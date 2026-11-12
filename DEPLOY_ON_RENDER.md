# Free public deployment on Render

This repository is configured for Render's **Free** web service. It provides a public `onrender.com` URL without a monthly hosting charge, subject to Render's free-plan limits: 512 MB RAM, automatic sleep after 15 minutes without traffic, temporary local files, and shared monthly usage limits. The first visit after sleep can take about a minute. Generated result files may disappear after restart or sleep.

## 1. Create Earth Engine credentials for the server

The website can load without these credentials, but place search/analysis requests to Earth Engine will fail until the service account is configured.

1. Open [Google Cloud Console](https://console.cloud.google.com/) and select project `satellite-change-ai`.
2. Confirm that the project is registered for **noncommercial** Earth Engine use and the Earth Engine API is enabled. Noncommercial Earth Engine registration itself does not require billing. Do not enable paid services for this demo.
3. In **IAM & Admin → Service Accounts**, create an account such as `terrascope-render`.
4. Grant it the Earth Engine Resource Viewer role (`roles/earthengine.viewer`). If Earth Engine reports a service-usage permission error, also grant Service Usage Consumer (`roles/serviceusage.serviceUsageConsumer`).
5. Create a JSON key for this service account and keep the downloaded file private. You will paste its contents into Render as a secret in step 3. Never commit this key to GitHub or send it in chat.

Google's service-account setup guide: <https://developers.google.com/earth-engine/guides/service_account>.

## 2. Create the Render service

1. Sign in at [Render](https://render.com/) with the GitHub account that can access `Lokeshconnected/SATELLITE-LAND-CHANGE-DETECTION`.
2. Choose **New → Blueprint** and select that repository and the `main` branch.
3. Render reads `render.yaml`. Review that the service plan says **Free**, then create the service.
4. The first build installs the Python requirements and may take several minutes. Wait for the deploy to finish.

## 3. Add the Earth Engine key as a secret

1. Open the new service in Render and choose **Environment**.
2. Add `EARTH_ENGINE_SERVICE_ACCOUNT_JSON` and paste the complete contents of the downloaded JSON key into its value. Mark/store it as a secret if Render offers that choice.
3. Save and redeploy.

Do not add the JSON key to GitHub, `render.yaml`, `README.md`, or a screenshot. If the key is ever exposed, delete it in Google Cloud and create a new one.

## 4. Open and check the site

Render shows the public URL in the service overview, for example `https://your-service.onrender.com`.

- Open the root URL for the site.
- Open `/api/health`; it should return `{"status":"ok","model":"loaded"}`.
- Run a small analysis using a location and date range with available Sentinel-2 imagery.

The free service sleeps after inactivity and has only 512 MB RAM. If logs show the process being killed for memory during an analysis, the trained-model workflow may exceed the free instance limit; the page can still load, but that analysis will need a host with more memory. Render's free local disk is temporary, so exported results are not permanent archives.

## Google Search

After the service is live, submit the public URL in [Google Search Console](https://search.google.com/search-console) and submit `https://your-service.onrender.com/sitemap.xml`. Google may take time to index it; neither indexing nor a top search position can be guaranteed. Render automatically serves a restrictive `robots.txt` while a free service is asleep, which can slow crawling.
