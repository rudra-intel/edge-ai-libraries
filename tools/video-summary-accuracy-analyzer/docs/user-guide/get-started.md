# Get Started

This guide provides step-by-step instructions to deploy and test the **Video Summary Accuracy Analyzer**. The application compares machine-generated video summaries with ground-truth references using BERTScore, ROUGE, semantic similarity, factual consistency, and temporal coherence metrics.

## Prerequisites

Before you begin, confirm the following:

- A Linux system with enough disk space for the container images and downloaded evaluation models.
- [Docker Engine](https://docs.docker.com/engine/install/) installed and running.
- [Docker Compose](https://docs.docker.com/compose/install/) plugin installed.
- Git and internet access to build the images and download the models on the first run.

Verify the Docker installation before continuing:

```bash
docker --version
docker compose version
```

## Configure the Environment

Run the setup script from the project root:

```bash
export HUGGINGFACEHUB_API_TOKEN=<your_huggingfacehub_token>
source scripts/setup_env.sh
```

The script creates the model cache and exports the values required by Docker Compose:

| Variable | Value | Purpose |
| --- | --- | --- |
| `MODEL_CACHE_PATH` | `~/model_cache/sbert` | Persists downloaded evaluation models on the host. |
| `SBERT_MODEL_ID` | `all-mpnet-base-v2` | Selects the Sentence-BERT model used for semantic similarity. |
| `USER_GROUP_ID` | Current user's primary group ID | Grants the backend container access to the mounted model cache. |
| `APP_BACKEND_URL` | `http://vss-acc-eval:9000/v1/eval` | Connects the dashboard to the backend service. |

The Compose configuration also forwards the host's `http_proxy`, `https_proxy`, and `no_proxy` values when they are set.

## Build the Application

If the `vss-eval:latest` and `vss-eval-ui:latest` images have not already been built, follow [How to Build from Source](./build-from-source.md). The quickest option from the project root is:

```bash
docker compose -f docker/compose.yaml build
```

## Run the Application with Docker Compose

Run the following commands from the project root:

1. Start the services in detached mode:

	```bash
	docker compose -f docker/compose.yaml up -d
	```

	Add `--build` to rebuild the images before starting the services:

	```bash
	docker compose -f docker/compose.yaml up -d --build
	```

2. Confirm that the services are running:

	```bash
	docker compose -f docker/compose.yaml ps
	```

	On the first run, startup can take several minutes while the evaluation models are downloaded to the model cache.

## Access and Use the Application

Open the dashboard in a browser:

```text
http://<host-ip>:8101
```

Use `http://localhost:8101` when the browser is running on the host. For remote access, replace `<host-ip>` with the IP address or hostname of the system running the containers and ensure that TCP port `8101` is reachable.

Interactive REST API documentation is available through the application gateway:

- Swagger UI: `http://localhost:8101/api/docs`
- OpenAPI schema: `http://localhost:8101/api/openapi.json`

For remote access, replace `localhost` with the host IP address or hostname.

The dashboard provides two workflows:

- **Summary Evaluation:** Upload one Markdown (`.md`) file containing `Reference` and `Generated` sections, then select **Submit** to view sentence comparisons, factual consistency, and temporal coherence results.
- **Metrics:** Enter reference and generated text, select a metric, and select **Submit** to calculate its score.

Use the following format for a Summary Evaluation input file:

```markdown
## Reference
The ground-truth summary goes here.

## Generated
The machine-generated summary goes here.
```

## Verify the Deployment

Check the backend health endpoint through the application gateway:

```bash
curl http://localhost:8101/api/health
```

A healthy service returns:

```json
{"status":"Success","message":"Service is up and running."}
```

Test an individual metric through the REST API:

```bash
curl --request POST http://localhost:8101/api/semantic-score \
	--header 'Content-Type: application/json' \
	--data '{
		"reference": "A person enters the room.",
		"generated": "Someone walks into the room."
	}'
```

The response includes the reference text, generated text, and semantic similarity score.

## Stop the Application

Stop and remove the application containers and network:

```bash
docker compose -f docker/compose.yaml down
```

The downloaded models remain in `~/model_cache/sbert` and are reused on subsequent runs.

## Troubleshooting

View the service logs if a container is unhealthy or the dashboard is unavailable:

```bash
docker compose -f docker/compose.yaml logs -f
```

To inspect one service only, specify `nginx`, `vss-acc-eval`, or `vss-acc-eval-ui` after `logs -f`.

- If port `8101` is already in use, stop the conflicting process or change the host-side port in `docker/compose.yaml`.
- If a model download fails, verify internet and proxy access, then restart the services.
- If the setup script cannot create or repair the model cache, ensure your user can write to `~/model_cache` and can run the script's required `sudo` command when an existing cache is owned by `root`.
- If remote clients cannot open the dashboard, allow inbound TCP traffic on port `8101` in the host firewall.

## Supporting Resources

- [How to Build from Source](./build-from-source.md)
- [Project README](../../README.md)

