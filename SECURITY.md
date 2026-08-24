# Security Policy

## Reporting a vulnerability

Please do **not** open a public GitHub issue for a vulnerability that could expose data, credentials, private network services, or arbitrary server-side access.

Use GitHub's private vulnerability reporting for this repository when available. If that is not available, contact the repository owner privately through the contact information on the GitHub profile.

Include reproduction steps, affected versions/commits, impact, and any suggested mitigation if known.

## Endpoint security model

KnowledgeLens makes server-side requests to user-configured OpenAI-compatible endpoints. Because a public deployment could otherwise become an SSRF primitive:

- public HTTPS endpoints are the secure default;
- local/loopback destinations require `KNOWLEDGELENS_ALLOW_LOCAL_ENDPOINTS=1`;
- private/link-local destinations require `KNOWLEDGELENS_ALLOW_PRIVATE_ENDPOINTS=1`;
- redirects are rejected rather than followed with credentials.

These opt-ins are intended for trusted self-hosted deployments, not shared public instances.

## Supported versions

Security fixes target the latest code on `main` until formal release support windows are introduced.
