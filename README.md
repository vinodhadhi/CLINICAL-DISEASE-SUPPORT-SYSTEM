# Project Structure

## Frontend

GitHub Pages files are in `frontend/`:

- `frontend/index.html`
- `frontend/script.js`
- `frontend/style.css`

## Backend

Render backend files are in `backend/`:

- `backend/app.py`
- `backend/requirements.txt`
- `backend/Dockerfile`
- `backend/disease_dataset (1).csv`

Render uses the root `render.yaml`, which points to `backend/Dockerfile`.

## Render deployment

The backend is set up for Render as a Docker web service.

1. Push this project to GitHub.
2. In Render, create a new Blueprint instance or Web Service from the repo.
3. Render will detect `render.yaml` at the repo root and build from `backend/Dockerfile`.
4. After deploy, your API health endpoint will be available at `/health`.

The frontend currently expects the production backend URL in `frontend/script.js`:

- Replace `https://your-render-backend.onrender.com` with your actual Render service URL.
