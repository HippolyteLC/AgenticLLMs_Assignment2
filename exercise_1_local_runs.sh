#!/bin/bash

# Define your Ollama base URL (change if hosted elsewhere, e.g., http://localhost:11434/v1)
URL="http://localhost:11434/v1"

# List of models you want to iterate over
MODELS=(
    "qwen2.5-3b-instruct-q4_k_m:latest"
    "smollm2-1.7b-instruct-q4km:latest"
    "SmolLM2-135M-Instruct-Q4:latest"
    "qwen3.5-0.8b-Q4:latest"
)

# Iterate through each model in the list
for MODEL_ID in "${MODELS[@]}"; do
    echo "=================================================="
    echo "Starting evaluation for model: ${MODEL_ID}"
    echo "=================================================="

    python eval.py \
      client.client_name=ollama \
      client.model_id="${MODEL_ID}" \
      eval.num_workers=4 \
      agent.type=naive \
      client.base_url="${URL}"

    echo "Completed evaluation for ${MODEL_ID}"
    echo ""
done

echo "All evaluations finished!"
