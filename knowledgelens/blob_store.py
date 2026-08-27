from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True, slots=True)
class BlobRecord:
    sha256: str
    byte_size: int
    ref: str


class LocalContentStore:
    """Streaming SHA-256-addressed local blob storage."""

    def __init__(self, root: str | os.PathLike[str] | None = None):
        configured = root or os.getenv("KNOWLEDGELENS_BLOB_ROOT", "data/blobs")
        self.root = Path(configured).expanduser().resolve()
        self.sha_root = self.root / "sha256"

    def _path_for(self, digest: str) -> Path:
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("Invalid SHA-256 blob identifier.")
        return self.sha_root / digest[:2] / digest

    def put_bytes(self, payload: bytes) -> BlobRecord:
        from io import BytesIO

        return self.put_stream(BytesIO(payload))

    def put_stream(self, stream: BinaryIO, chunk_size: int = 1024 * 1024) -> BlobRecord:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        self.root.mkdir(parents=True, exist_ok=True)
        self.sha_root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass

        hasher = hashlib.sha256()
        total = 0
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, prefix=".blob-", delete=False) as temp:
                temp_name = temp.name
                while True:
                    chunk = stream.read(chunk_size)
                    if not chunk:
                        break
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise TypeError("Blob streams must return bytes.")
                    raw = bytes(chunk)
                    hasher.update(raw)
                    total += len(raw)
                    temp.write(raw)

            digest = hasher.hexdigest()
            destination = self._path_for(digest)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                os.unlink(temp_name)
            else:
                os.replace(temp_name, destination)
                try:
                    os.chmod(destination, 0o600)
                except OSError:
                    pass
            return BlobRecord(sha256=digest, byte_size=total, ref=f"sha256:{digest}")
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

    def open(self, ref: str) -> BinaryIO:
        digest = self.digest_from_ref(ref)
        return self._path_for(digest).open("rb")

    def exists(self, ref: str) -> bool:
        return self._path_for(self.digest_from_ref(ref)).is_file()

    @staticmethod
    def digest_from_ref(ref: str) -> str:
        prefix = "sha256:"
        if not ref.startswith(prefix):
            raise ValueError("Unsupported blob reference.")
        digest = ref[len(prefix):]
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("Invalid SHA-256 blob reference.")
        return digest
