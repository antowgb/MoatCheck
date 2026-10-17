# Supabase Auth setup

MoatCheck uses one Supabase project for the application. Users sign up with an email and password in MoatCheck. They do not need their own Supabase account or project.

## 1. Frontend settings

Set these values in `frontend/.env.local` for local development and in the frontend build environment for production:

```env
NEXT_PUBLIC_SUPABASE_URL=https://<project-ref>.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=<publishable-or-anon-key>
NEXT_PUBLIC_SITE_URL=https://<canonical-app-origin>
```

The legacy `NEXT_PUBLIC_SUPABASE_ANON_KEY` is supported if the publishable key is not available. These are browser credentials. Never use `SUPABASE_SECRET_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, a database password, or any other server secret in a `NEXT_PUBLIC_*` variable.

`NEXT_PUBLIC_SITE_URL` is optional in local development. Set it in the production frontend build to the canonical public origin so confirmation and password-reset emails return to the deployed app.

## 2. Supabase Auth dashboard

1. In **Authentication → Sign In / Providers**, enable Email and password sign-in. Choose whether new accounts require email confirmation.
2. In **Authentication → URL Configuration**, set the deployed site URL and add exact redirect URLs for each environment, for example:
   - `http://localhost:3000/auth/callback`
   - `http://localhost:3001/auth/callback` if using port 3001
   - `https://<your-domain>/auth/callback`
3. Ensure the confirmation and password-reset templates use Supabase's `{{ .ConfirmationURL }}` link.
4. If email confirmation is enabled, users must open the message and finish the callback before signing in.

Changing the site URL does not change links already sent. Request a fresh confirmation or reset email after changing redirect configuration.

## 3. Backend settings

Configure the same project URL on the backend along with:

- `SUPABASE_KEY`: publishable/legacy anon key for public research-data access.
- `SUPABASE_SECRET_KEY`: server-only elevated key for authenticated repository operations and user verification. `SUPABASE_SERVICE_ROLE_KEY` is supported as a legacy fallback.
- `APP_ENV=production` and `COOKIE_SECURE=true` when served over HTTPS.
- `FRONTEND_ORIGINS`: exact comma-separated origins only if the browser frontend is hosted separately. For the default same-origin Render deployment, leave it unset.

The elevated credential bypasses RLS. Keep it only in backend/hosting secrets. The backend verifies the Supabase user and uses repository owner checks before reading or writing a user's saved data.

Apply `backend/supabase_migration_security_hardening.sql` after the portfolio tracking, quant persistence, and Advisor migrations. It adds explicit owner columns and RLS policies to user portfolio child rows and restores service-role-only execution on account-wide quantitative data deletion.

## Session behavior

The browser Supabase client handles sign-in, signup, and reset. After authentication, the frontend sends the access token to `POST /api/v1/session`. The API verifies it and sets a one-hour `moatcheck_session` cookie with `HttpOnly`, `SameSite=Lax`, API-path scope, and `Secure` in production. JavaScript cannot read this API cookie.

The browser Supabase SDK keeps its short-lived login session in memory only; it is not written to `localStorage` or `sessionStorage`. After each page load, the UI reads the identity from the cookie-backed `GET /api/v1/session` endpoint. The API cookie is `HttpOnly`, so browser JavaScript cannot read the API credential. On cookie-authenticated state-changing API requests, the frontend sends `X-MoatCheck-Request: 1`; the backend also checks Origin or Referer against the same host and the exact configured `FRONTEND_ORIGINS` list.

To sign out, the frontend clears the API cookie and signs out from the Supabase browser client. The cookie is short-lived and expires after one hour. Users may need to sign in again after expiry.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `Invalid login credentials` | Confirm the user exists in this Supabase project, the email is confirmed if confirmation is required, and frontend URL/key point to the same project. |
| No confirmation/reset email | Check Supabase Auth email provider status, spam, rate limits, and the configured SMTP/email provider. |
| Email opens localhost | Set `NEXT_PUBLIC_SITE_URL` in the frontend build environment, add its callback URL to Supabase's redirect allowlist, rebuild/redeploy, then request a new email. |
| Browser CSP blocks Supabase Auth | The production API builds `connect-src` from the backend `SUPABASE_URL`. Ensure it is the correct HTTPS project URL and rebuild/redeploy. |
| API returns 401 after login | Confirm `/api/v1/session` succeeded, browser cookies are allowed, frontend and API share the expected origin, and session was not expired. |
| API returns 403 on a write | Check that the frontend sent `X-MoatCheck-Request: 1`, sent Origin or Referer, and any separate origin is listed exactly in `FRONTEND_ORIGINS`. |
| Local API requests fail | Set `NEXT_PUBLIC_API_URL=http://localhost:8000`, run FastAPI on port 8000, and restart Next.js after changing environment files. |

Never share access tokens, Supabase secret keys, service-role keys, or database passwords in screenshots, issues, or chat logs.
