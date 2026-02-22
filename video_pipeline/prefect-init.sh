#!/usr/bin/env bash
set -e

echo "Waiting for Prefect API..."
until python - <<EOF
import urllib.request
urllib.request.urlopen("http://prefect-server:4200/api/health")
EOF
do
  sleep 2
done

echo "Prefect API ready"
echo "Ensuring work pool exists..."
prefect work-pool create --type process local-pool || true

echo "Ensuring concurrency limits..."
prefect concurrency-limit create llm-service 3 || true
prefect concurrency-limit create embedding-service 3 || true
prefect concurrency-limit create autoshot-task 20 || true
# prefect concurrency-limit create asr-task 20 || true
prefect concurrency-limit create video-registry 1 || true

echo "Deploying flows..."
cd /app
prefect --no-prompt deploy --all


echo "Prefect initialization complete"


