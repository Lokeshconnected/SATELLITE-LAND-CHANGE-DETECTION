# Deploy the Website Online

This project can be deployed as a Render Web Service connected to the GitHub repository.

## Important security rule

Do not upload an Earth Engine credentials file to GitHub. The online server must receive the service-account JSON through Render's private environment-variable field.

## Before creating the Render service

Create a Google Cloud service account in project `satellite-change-ai-507112`. Give that service account access to Earth Engine and the project roles required by Earth Engine. Register the project for Earth Engine if it is not already registered.

Create a JSON key for the service account and keep the downloaded file only on your computer. The JSON should contain fields such as `client_email`, `private_key`, and `project_id`.

## Render settings

Create a new **Web Service** from:

```text
https://github.com/Lokeshconnected/SATELLITE-LAND-CHANGE-DETECTION
```

Use these settings if Render asks for them:

```text
Branch: main
Build command: pip install -r requirements.txt
Start command: uvicorn app:app --host 0.0.0.0 --port $PORT
Health check path: /api/health
```

Add these environment variables in Render. Use **Secret** for the JSON value:

```text
EARTH_ENGINE_PROJECT = satellite-change-ai-507112
EARTH_ENGINE_SERVICE_ACCOUNT_JSON = paste the complete service-account JSON text here
```

Do not put the JSON in `render.yaml`, `app.py`, GitHub, screenshots, or the paper.

## After deployment

Render will provide a public URL ending in `onrender.com`. Open:

```text
https://your-service-name.onrender.com/api/health
```

Expected response:

```json
{"status":"ok","model":"loaded"}
```

Then open the service root URL and run a place analysis.

## Free-plan behavior

The free service may sleep after inactivity, so the first request can be slow. Generated files are temporary and may disappear after a restart. That is acceptable for a college demonstration because each analysis creates its files again.

## If deployment fails

- Check the Render build logs for package-install errors.
- Check that the start command uses `$PORT`.
- Check that `EARTH_ENGINE_PROJECT` is the numeric-suffix project ID, not only the display name.
- Check that the service-account JSON is complete and valid JSON.
- Check that the service account has Earth Engine access.
- Check that the Earth Engine API is enabled and the project is registered.
- Do not paste the private key into a public issue or chat.
