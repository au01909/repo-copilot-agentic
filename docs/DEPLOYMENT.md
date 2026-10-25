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

## Free hosting options

Everything below is from each vendor's published pricing/docs at the time of
writing and was **not** tried with this app; check the live pages before relying
on a number. The LLM itself is separate: an NVIDIA API key from
[build.nvidia.com](https://build.nvidia.com) is the free option the app supports,
and it is rate limited (see the README).

| Piece | Option | What to know |
|---|---|---|
| Backend | **Google Cloud Run** | Free allowance under request-based billing: 180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests a month ([pricing](https://cloud.google.com/run/pricing)). A billing account must be attached even though small use stays free. At 1 vCPU / 1 GiB that is roughly 50 hours of active instance time a month; at 2 vCPU / 2 GiB about 25. `--min-instances=0` so idle time costs nothing. |
| Backend | **Hugging Face Spaces (Docker)** | The free CPU Basic tier is listed at 2 vCPU / 16 GB and sleeps after a period of inactivity ([docs](https://huggingface.co/docs/hub/spaces-overview)); forum reports mention limits on concurrent free Spaces. Needs the container to listen on the Space's `app_port` (7860 by default), so the Dockerfile's port and `CMD` would need adjusting. No persistent disk on the free tier. |
| Frontend | GitHub Pages, Netlify or Vercel | It is a single static file. Set the API address box to the backend URL and set the backend's `CORS_ORIGINS` to the page's origin. The Architecture diagram loads Mermaid from a CDN. |

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

echo -n "$NVIDIA_NIM_API_KEY" | gcloud secrets create nvidia-api-key --data-file=-
# optional, only if you use those providers instead:
#   ANTHROPIC_API_KEY / OPENAI_API_KEY / COHERE_API_KEY / GITHUB_TOKEN
# The app reads the NVIDIA key from NVIDIA_NIM_API_KEY, so that is the name to
# expose it under in step 3 (a secret mounted as NVIDIA_API_KEY is ignored).
```

## 3. Deploy

```bash
gcloud run deploy repo-copilot-backend \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/backend:latest \
  --region=$REGION \
  --platform=managed \
  --allow-unauthenticated \
  --port=8000 \
  --memory=1Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=1 \
  --timeout=300 \
  --set-env-vars="LLM_PROVIDER=nvidia,NVIDIA_MODEL=nvidia/nemotron-3-ultra-550b-a55b,AGENT_FRAMEWORK=direct,NVIDIA_RPM_LIMIT=10,ENABLE_MULTI_QUERY=false,VECTOR_STORE=qdrant_memory,SESSION_STORE=memory,ENABLE_MLFLOW=false,CORS_ORIGINS=https://YOUR-FRONTEND-ORIGIN" \
  --set-secrets="NVIDIA_NIM_API_KEY=nvidia-api-key:latest"
```

What the settings are for: `NVIDIA_MODEL` must be a model your key can still call (the
code's built-in default is retired and answers HTTP 410); `AGENT_FRAMEWORK=direct`
skips the ADK agent, which takes several model turns per answer; `NVIDIA_RPM_LIMIT=10`
keeps requests under the free-tier rate (see the README's
[rate-limit notes](../README.md#agent-cost-controls-and-nvidia-rate-limits)); and
`CORS_ORIGINS` should be your frontend's origin rather than the default `*`.
Use `AGENT_FRAMEWORK=adk` if you want Agent mode, and expect 30 s to over a minute per
agent answer at that rate.

Notes on the flags:
- `--allow-unauthenticated` is the simplest starting point for a portfolio
  deployment; switch to IAM-based auth (`--no-allow-unauthenticated` + a
  frontend that attaches an identity token) once this is more than a demo.
- `--min-instances=0` means cold starts (a few seconds) but zero cost when
  idle — appropriate for a low-traffic portfolio deployment. Set to `1` if
  cold-start latency matters more than idle cost.
- `SESSION_STORE=memory` and `VECTOR_STORE=qdrant_memory` both mean **state is
  lost on every container restart/cold-start-to-a-new-instance**, since Cloud
  Run's filesystem is ephemeral and instances aren't guaranteed to be reused.
  (SQLite on the container's disk has the same problem.) A session that outlives
  its process is rebuilt by re-indexing, but only if the session store still has
  it, so with these settings an old `session_id` after a cold start is a 404 and
  the user indexes again. `--max-instances=1` keeps one session on one instance;
  with more, a request can land on an instance that never indexed the repo.
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
