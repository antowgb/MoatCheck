# Security and responsible disclosure

MoatCheck is a public educational research application. It does not execute trades or connect to brokerage accounts. Public market-research endpoints and authenticated user-data endpoints have different access rules.

## Reporting a vulnerability

Please do not post tokens, API keys, personal data, or exploit details in a public issue. Use GitHub's private vulnerability reporting for this repository if enabled. Otherwise contact the maintainer privately before sharing details publicly. Include the affected route/component, impact, and a minimal reproduction without live credentials.

## Secrets and deployment

- Keep Supabase secret/service-role keys, database credentials, provider keys, admin keys, and email provider credentials in backend/hosting secrets only.
- Browser settings may contain only the Supabase project URL and publishable/anon key. Never put an elevated Supabase key in a `NEXT_PUBLIC_*` variable.
- `ADMIN_API_KEY` must be a strong random value. Restricted admin writes fail closed if it is not configured.
- The default production deployment serves the static frontend and API from one origin. If a separate frontend origin is introduced, configure `FRONTEND_ORIGINS` with exact trusted origins. Do not use `*` with credentials.
- Production builds use restrictive CSP/security headers. The Supabase HTTPS origin must be configured on the backend so browser Auth can connect under CSP.

## Authentication and owner data

Supabase Auth verifies user identity. The frontend exchanges a verified access token for a short-lived HttpOnly API cookie. Cookie-authenticated state-changing requests require the `X-MoatCheck-Request: 1` proof header; CORS is restricted to configured origins. The browser Supabase SDK also retains its own session storage, so not every auth credential is exclusively HttpOnly.

The elevated Supabase client bypasses RLS. API repositories therefore apply explicit owner scoping using the verified user ID. Database RLS and service-only function grants are defense in depth and must be reviewed when migrations change. The data-deletion endpoint removes saved quant records covered by the privacy SQL routine; it does not remove the Supabase Auth identity or already sent email.

## Request protections and operational limits

The API sets a request-body limit, a per-client fixed-window rate limit, CSRF origin proof for cookie writes, and browser security headers. The current limiter is held in application memory. It protects a single process from basic request bursts; it is not a distributed denial-of-service control. Use a shared limiter and hosting edge controls before scaling to multiple workers/instances. Trust forwarded IP headers only when a trusted proxy overwrites them.

The data pipeline contacts third-party providers and uses configured credentials. API keys do not grant data storage or public redistribution rights. Review each provider contract before enabling persistent data writes.

## User data and public portfolio

Persisted portfolio and recommendation data is private to the authenticated owner through application-level checks. Do not enter brokerage passwords, payment data, or sensitive information that the application does not need. Review the in-app Privacy page before saving data.

## Scope of this document

These notes describe implemented controls and known limits. They are not a penetration test, formal security certification, guarantee against attack, or proof that the live hosting/database configuration matches the repository. Validate deployed secrets, Supabase RLS policies, RPC grants, CORS origins, and email redirects in their respective consoles. The latest code audit lists remaining checks in [the platform audit](docs/platform-audit-2026-10-17.md).
