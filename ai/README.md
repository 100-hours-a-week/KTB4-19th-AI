# zipsai API

## Langfuse tracing

Create a Langfuse project and set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and
`LANGFUSE_BASE_URL` in the root `.env` (or deployment environment). Use the URL
for the region where the project was created. Both keys enable tracing; set
`LANGFUSE_TRACING_ENABLED=false` to turn it off.
Set `LANGFUSE_TRACING_ENVIRONMENT` to `development`, `staging`, or `production`
for the deployment.

Each `/converse` turn is one trace, grouped by `conversation_id`. Existing stage
timings and LLM/VLM generations appear beneath it. Model and token usage are
recorded, while prompt text, replies, and image URLs are removed before export.
The root records only message presence, image count, history length, status,
route, and `turn_id`.

Each accepted `/indexing/jobs` background job creates one trace with its setup,
indexing, and reconciliation stages. Its trace records only job status and counts;
document titles, file keys, and content are excluded.

On macOS, if OTLP export reports a certificate verification error, set
`OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE=/etc/ssl/cert.pem` in your local environment.

For production, set the Langfuse keys and region URL in `/opt/zipsai/.env`, and
pass them to both `ai-api` and `embedding` in `/opt/zipsai/compose.yaml`. The CD
workflow deploys images but does not copy the repository's Compose file to EC2.
Deploy both image tags after updating that server configuration.
