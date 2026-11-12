# Satellite Land Change Detection AI

An interactive satellite analysis app built around the trained U-Net model in this repository. Search a place, compare Sentinel-2 imagery across years, inspect change overlays and region summaries, and export analysis results.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
earthengine authenticate
export EARTH_ENGINE_PROJECT=satellite-change-ai
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>. Place-based analysis requires Google Earth Engine access. For service-account deployment, configure `EARTH_ENGINE_SERVICE_ACCOUNT_JSON` and `EARTH_ENGINE_PROJECT` as environment variables. Do not commit service-account keys.

## Deploy

The repository includes a `render.yaml` blueprint for a free Render web service and a step-by-step [deployment guide](DEPLOY_ON_RENDER.md). The free host sleeps when idle and has limited memory; analysis also requires a private Earth Engine service-account key configured in Render.

## Search indexing

The app serves `robots.txt` and `sitemap.xml` for its public host, plus page metadata and structured data. After deploying to a public URL, add the site to Google Search Console and submit `/sitemap.xml`. Search indexing and rankings are determined by Google and are not guaranteed.

Analysis outputs are screening signals for review, not certified surveys or ground truth. Cloud cover, season, illumination, and acquisition conditions can affect results.
