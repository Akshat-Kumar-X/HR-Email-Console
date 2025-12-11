# HR Email Console

A minimal, beautiful **HR Email Console** for managing pre-boarding / onboarding emails.

<img width="2100" height="4217" alt="image" src="https://github.com/user-attachments/assets/61f89dfa-367f-4af2-a07a-1e2a0208132e" />

## Tech Stack

**Backend**

- Python 3.10+
- FastAPI
- SQLAlchemy + SQLite
- Background worker using `asyncio`

**Frontend**

- Single-page HTML + vanilla JS
- Dark **black / blue** UI

---

## Project Structure

```text
.
├── backend/
│   ├── main.py               # FastAPI app, models, job worker, REST API
│   └── gmail_service.py      # (optional) Gmail API via service account
└── frontend/
    └── index.html            # Minimal HR Email Console UI
