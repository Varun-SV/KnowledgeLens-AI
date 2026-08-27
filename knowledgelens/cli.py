from __future__ import annotations

import argparse
import getpass
import sys

from .auth import bootstrap_admin
from .database import Database, DatabaseUnavailable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="knowledgelens", description="KnowledgeLens administration utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    bootstrap = subparsers.add_parser(
        "bootstrap-admin",
        help="Create the first local administrator in the configured PostgreSQL database.",
    )
    bootstrap.add_argument("--username", required=True, help="Administrator username (1-120 characters).")
    return parser


def _bootstrap_admin_command(username: str) -> int:
    database = Database()
    if not database.enabled:
        print("KNOWLEDGELENS_DATABASE_URL is required before bootstrapping an administrator.", file=sys.stderr)
        return 2

    try:
        database.initialize()
        password = getpass.getpass("Administrator password: ")
        confirmation = getpass.getpass("Confirm administrator password: ")
        if password != confirmation:
            print("Passwords do not match.", file=sys.stderr)
            return 2
        created = bootstrap_admin(database, username, password)
    except (DatabaseUnavailable, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not created:
        print("An administrator already exists; bootstrap-admin made no changes.", file=sys.stderr)
        return 1
    print(f"Created bootstrap administrator: {username.strip()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "bootstrap-admin":
        return _bootstrap_admin_command(args.username)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
