# Optional Gmail newsletter collector

The qualitative pipeline can read messages from a manually curated sender allow-list to extract ticker events. Gmail support is disabled by default in `app/qualitative/config.py`. The default event sources do not require Gmail.

## Access scope

The collector requests only `https://www.googleapis.com/auth/gmail.readonly`. It does not send, draft, label, archive, or delete messages. Use a dedicated Google account/mailbox if possible and include only senders whose newsletters you intend to process.

## Local setup

1. In Google Cloud Console, create a project, enable the Gmail API, and create an OAuth client of type **Desktop app**.
2. Save the downloaded OAuth client JSON as `backend/gmail_credentials.json`. This file is gitignored. Set `GMAIL_API_CREDENTIALS_PATH` in `backend/.env` to its path.
3. From the `backend/` directory with the virtual environment active, run the one-time local consent flow below. It opens a browser and writes `gmail_token.json` next to the credentials file.

```bash
cd backend
./.venv/bin/python - <<'PY'
import os
from pathlib import Path
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from app.qualitative.collectors.gmail_newsletter import GMAIL_SCOPES

load_dotenv()
credentials_path = Path(os.environ["GMAIL_API_CREDENTIALS_PATH"]).expanduser()
flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), GMAIL_SCOPES)
credentials = flow.run_local_server(port=0)
token_path = credentials_path.with_name("gmail_token.json")
token_path.write_text(credentials.to_json(), encoding="utf-8")
print(f"Wrote token file: {token_path}")
PY
```

The command loads local settings through `python-dotenv`. Do not paste `.env` contents into an issue, commit, or chat. The token grants read access to the mailbox and must remain private.

4. Copy `backend/newsletter_senders.example.json` to the path configured by `NEWSLETTER_SENDERS_FILE` (default `backend/qualitative_cache/newsletter_senders.json`) and add the sender addresses you want to allow.
5. Enable the newsletter source in `app/qualitative/config.py` only after credentials and the sender allow-list are ready.

## Scheduled runs

An unattended runner needs the OAuth client file and refresh token provisioned securely. The repository's scheduled data workflow does not provision a Gmail token by default. Do not enable the newsletter source in CI until secret handling, token refresh, and revocation procedures are configured.

If credentials are absent or invalid, the collector records source health as failed/unavailable rather than making newsletter data a required input for numeric scoring. Qualitative event labels remain separate from fundamental, risk, and composite scores.

## Revoking access

Revoke the OAuth grant in the Google account's connected-app settings, then remove local `gmail_credentials.json` and `gmail_token.json`. Deleting those local files does not revoke the grant by itself.
