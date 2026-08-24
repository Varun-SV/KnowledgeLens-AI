# Security Policy

## Reporting a vulnerability

Please do **not** open a public GitHub issue for a vulnerability that could expose data, credentials, private network services, or arbitrary server-side access.

Use GitHub's private vulnerability reporting for this repository when available. If that is not available, contact the repository owner privately through the contact information on the GitHub profile.

Include reproduction steps, affected versions/commits, impact, and any suggested mitigation if known.

## Endpoint security model

KnowledgeLens makes server-side requests to OpenAI-compatible endpoints, so endpoint selection is intentionally operator-controlled:

- public visitors select only from built-in providers or an operator-configured endpoint;
- arbitrary custom endpoints are configured with `KNOWLEDGELENS_CUSTOM_ENDPOINT`, not typed into the public UI;
- public destinations must use HTTPS;
- local/loopback destinations require `KNOWLEDGELENS_ALLOW_LOCAL_ENDPOINTS=1`;
- private/link-local destinations require `KNOWLEDGELENS_ALLOW_PRIVATE_ENDPOINTS=1`;
- local/private opt-ins do **not** permit plaintext HTTP to public addresses;
- unspecified, multicast, and reserved destinations are rejected;
- redirects are rejected rather than followed with credentials.

These opt-ins are intended for trusted self-hosted deployments, not shared public instances.

## Supported versions

Security fixes target the latest code on `main` until formal release support windows are introduced.
