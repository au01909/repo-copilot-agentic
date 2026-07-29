# Deploying to Google Cloud Run

> **Verification status:** these commands were written and reviewed against
> Cloud Run's documented CLI interface, but **not executed** — the environment
> this was built in has no GCP credentials and no network access to any
> `*.googleapis.com` endpoint (only PyPI/npm/GitHub are reachable there). Test
> this against a real GCP project before trusting it for a real deployment,
> and treat this as a well-researched starting point, not a verified runbook.

## Why Cloud Run, and what's deliberately *not* here

Per the platform's own design goal: **the LLM stays external** (NVIDIA AI
Endpoints via `ChatNVIDIA`/ADK, or Anthropic/OpenAI/DeepSeek via the existing
provider gateway) — nothing here deploys or hosts a model. Cloud Run only
needs to run the FastAPI container. Qdrant runs in `qdrant_memory` or
`qdrant_local` mode inside that same container for a low-cost deployment
(no separate Qdrant service) — see the tradeoff note below if you want
persistent vector storage across container restarts, which Cloud Run's
ephemeral filesystem doesn't give you for free.

## Prerequisites

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

## 1. Build and push the image to Artifact Registry

```bash
export PROJECT_ID=your-project-id
export REGION=us-central1
export REPO=repo-copilot

gcloud artifacts repositories create $REPO \
  --repository-format=docker \
  --location=$REGION \
  --description="Repository Copilot images"

gcloud auth configure-docker ${REGION}-docker.pkg.dev

cd backend
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/backend:latest .
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/backend:latest
```

## 2. Store secrets (don't pass API keys as plain env vars in production)

```bash
gcloud services enable secretmanager.googleapis.com

echo -n "$ANTHROPIC_API_KEY" | gcloud secrets create anthropic-api-key --data-file=-
echo -n "$NVIDIA_API_KEY"    | gcloud secrets create nvidia-api-key --data-file=-
# repeat for OPENAI_API_KEY / COHERE_API_KEY / GITHUB_TOKEN / etc. as needed
```

## 3. Deploy

```bash
gcloud run deploy repo-copilot-backend \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/backend:latest \
  --region=$REGION \
  --platform=managed \
  --allow-unauthenticated \
  --port=8000 \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=0 \
  --max-instances=3 \
  --timeout=300 \
  --set-env-vars="LLM_PROVIDER=nvidia,VECTOR_STORE=qdrant_memory,SESSION_STORE=sqlite,ENABLE_MLFLOW=false" \
  --set-secrets="NVIDIA_API_KEY=nvidia-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest"
```

Notes on the flags:
- `--allow-unauthenticated` is the simplest starting point for a portfolio
  deployment; switch to IAM-based auth (`--no-allow-unauthenticated` + a
  frontend that attaches an identity token) once this is more than a demo.
- `--min-instances=0` means cold starts (a few seconds) but zero cost when
  idle — appropriate for a low-traffic portfolio deployment. Set to `1` if
  cold-start latency matters more than idle cost.
- `SESSION_STORE=sqlite` and `VECTOR_STORE=qdrant_memory` both mean **state is
  lost on every container restart/cold-start-to-a-new-instance**, since Cloud
  Run's filesystem is ephemeral and instances aren't guaranteed to be reused.
  For a portfolio demo where "index a repo, ask a few questions" happens
  within one session, this is usually fine. For anything persistent, see
  the upgrade path below.
- `ENABLE_MLFLOW=false` for the initial deployment — MLflow's SQLite backend
  has the same ephemeral-filesystem problem; wire it to a separate persistent
  store (or a managed MLflow instance) before relying on it in production.

## 4. Verify

```bash
export SERVICE_URL=$(gcloud run services describe repo-copilot-backend \
  --region=$REGION --format='value(status.url)')

curl "$SERVICE_URL/api/health"
curl "$SERVICE_URL/api/ready"
```

## 5. Point the frontend at it

Edit `frontend/index.html`'s `apiBase` field (or set it via the input box in
the UI) to `$SERVICE_URL` instead of `http://localhost:8000`. The frontend is
static — host it anywhere (Cloud Storage + a load balancer, Netlify, Vercel,
GitHub Pages); it only needs to reach the Cloud Run URL over HTTPS.

## Logs

```bash
gcloud run services logs read repo-copilot-backend --region=$REGION --limit=100
```

## Production upgrade path (documented, not required for a portfolio deployment)

These make the deployment more durable but add cost and operational surface —
skip them for a demo, consider them once this needs to stay up reliably:

| Concern | Low-cost default (above) | Upgrade |
|---|---|---|
| Vector storage persistence | `qdrant_memory` (lost on restart) | A managed/self-hosted Qdrant instance (Qdrant Cloud free tier, or a small Compute Engine VM) + `VECTOR_STORE=qdrant_server` |
| Session persistence | SQLite on ephemeral disk | Cloud SQL (Postgres) — would need `persistence.py`'s `SessionStore` interface implemented against it (it's adapter-shaped already, see `backend/app/persistence.py`) |
| Secrets | Secret Manager (already used above) | same — this one's already the right call, no upgrade needed |
| Frontend hosting | any static host | Cloud Storage bucket + Cloud CDN, or Firebase Hosting |
| Custom domain / HTTPS | Cloud Run's default `*.run.app` URL | `gcloud run domain-mappings create` + your own domain |
| Observability | Cloud Run's built-in request logs | Cloud Monitoring dashboards + the existing LangSmith/MLflow integrations pointed at persistent backends |
| CI/CD | manual `gcloud run deploy` | `.github/workflows/ci-cd.yml` in this repo already builds the image on every push to `main`; add a `gcloud run deploy` step once you have a service account key or Workload Identity Federation set up (see that file's comments) |

None of this is required to get the low-cost deployment above working — it's
here so the tradeoffs are explicit rather than silently absent.
