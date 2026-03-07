# Security & Secrets

Never commit credentials, API keys, or personal data to git. This project keeps them out of version control.

## What's Git-Ignored

| File / Path | Why |
|-------------|-----|
| `.env` | All API keys and secrets |
| `.env.local`, `.env.*.local` | Local overrides |
| `firestore.rules` | Generated file containing owner email |
| `frontend/dist` | Build output may inline env vars |
| `*-firebase-adminsdk-*.json` | Firebase service account keys |

## Firestore Rules & Owner Email

Your email is used in Firestore security rules so only you can access the app. It must **never** be committed.

1. Add `OWNER_EMAIL=you@example.com` to your `.env` file.
2. Before each deploy, run:
   ```bash
   node scripts/generate-firestore-rules.js
   ```
3. This generates `firestore.rules` from `firestore.rules.template` (the template is safe to commit).

If you previously committed `firestore.rules` with your real email:

```bash
git rm --cached firestore.rules
git commit -m "Stop tracking firestore.rules (contains email)"
```

Then consider [removing the email from git history](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository) if it was pushed.

## API Keys (Backend)

All backend secrets are loaded from `.env`:

- `OPENAI_API_KEY`, `NEWSAPI_KEY`, `FINNHUB_KEY`, `ALPHAVANTAGE_KEY`
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL`

Copy `.env.example` to `.env` and fill in your keys. `.env` is git-ignored.

## Frontend (Firebase)

Firebase config is loaded from `frontend/.env` via Vite:

- `VITE_FIREBASE_API_KEY`, `VITE_FIREBASE_AUTH_DOMAIN`, etc.

Copy `frontend/.env.example` to `frontend/.env`. Build output (`frontend/dist`) is git-ignored so built files with inlined values are not committed.
