# Deploying to Google Cloud Run

Follow these steps to deploy the Amazon Image Translator to GCP Cloud Run.

## Prerequisites

1.  **Google Cloud Platform Project**: Ensure you have a GCP project with billing enabled.
2.  **gcloud CLI**: Installed and authenticated (`gcloud auth login`).
    *   [Download and Install Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
    *   **Tip**: If you don't want to install anything, you can use **[Google Cloud Shell](https://shell.cloud.google.com)** (it has everything pre-installed).
3.  **APIs Enabled**:
    *   Cloud Run API
    *   Cloud Build API
    *   Artifact Registry API
    *   Cloud Vision API
    *   Cloud Translation API

## Deployment Steps

### 1. Set Project & Region
Run these in your terminal:

```powershell
gcloud config set project pivotal-robot-369519
gcloud config set run/region us-central1
```

### 2. Build and Submit Container
This builds your Docker container and stores it in Google Container Registry (GCR) or Artifact Registry.

```powershell
gcloud builds submit --tag gcr.io/pivotal-robot-369519/process-app
```

### 3. Deploy to Cloud Run
Deploy the service. We use `--allow-unauthenticated` so it's publicly accessible (optional).

```powershell
gcloud run deploy process-app `
  --image gcr.io/pivotal-robot-369519/process-app `
  --platform managed `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 1
```

### 4. Authentication (Crucial!)
The app needs permissions to use Vision and Translate APIs.

1.  **Create a Service Account** (if you haven't already):
    ```powershell
    gcloud iam service-accounts create amazon-translator-sa
    ```

2.  **Grant Permissions**:
    Give it access to **Cloud Vision** and **Cloud Translation**.
    ```powershell
    gcloud projects add-iam-policy-binding pivotal-robot-369519 `
      --member "serviceAccount:amazon-translator-sa@pivotal-robot-369519.iam.gserviceaccount.com" `
      --role "roles/cloudtranslate.user"

    gcloud projects add-iam-policy-binding pivotal-robot-369519 `
      --member "serviceAccount:amazon-translator-sa@pivotal-robot-369519.iam.gserviceaccount.com" `
      --role "roles/serviceusage.serviceUsageConsumer"
    ```
    *(Note: You might need to add specific roles for Vision API as well, typically contained in basic editor or specific `roles/vision.imageAnnotator` if available, or just verify the SA has access).*

3.  **Attach Service Account to Cloud Run Service**:
    ```powershell
    gcloud run services update process-app `
      --service-account amazon-translator-sa@pivotal-robot-369519.iam.gserviceaccount.com
    ```

## Troubleshooting
- **Logs**: View logs in the Cloud Run console to see startup errors.
- **504 Metadata Error**: This usually means the app is trying to reach the metadata server for credentials but failing. Ensure the Service Account is correctly attached.

## Custom Domain Mapping
To use your own domain (e.g., `app.example.com`) instead of the default `run.app` URL:

1.  **Go to Cloud Run Console**:
    - Open your service (`process-app`).
    - Click **Manage Custom Domains**.

2.  **Add Mapping**:
    - Click **Add Mapping**.
    - Select "Cloud Run Domain Mappings".
    - Select your verified domain (you may need to verify ownership via Webmaster Central if it's not listed).
    - Enter the subdomain (e.g., `www` or leave blank for root).

3.  **Update DNS Records**:
    - Google will provide you with DNS records (usually `A` and `AAAA` records).
    - Go to your domain registrar (GoDaddy, Namecheap, etc.) and add these records.
    - It may take up to 24 hours for the SSL certificate to be issued and the domain to work.
