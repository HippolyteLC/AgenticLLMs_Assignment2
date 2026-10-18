#!/bin/bash

# chmod +x [PATH], to enable running script

# Define your Ollama base URL (change if hosted elsewhere, e.g., http://localhost:11434/v1)
URL="http://localhost:11434/v1"

# List of models you want to iterate over
MODELS=(
    # "gemma-4-E2B-it-Q4_K_M:latest"
    # "qwen2.5-3b-instruct-q4_k_m:latest"
    "Llama-3.2-3B-Instruct-Q4_K_M:latest"
    # "smollm2-1.7b-instruct-q4km:latest"
)

# Iterate through each model in the list
for MODEL_ID in "${MODELS[@]}"; do
    echo "=================================================="
    echo "Starting evaluation for model: ${MODEL_ID}"
    echo "=================================================="

    python eval.py \
      client.client_name=ollama \
      client.model_id="${MODEL_ID}" \
      eval.num_workers=1 \
      agent.type=custom \
      client.base_url="${URL}" \
      envs.names=babyai
      #eval.num_episodes.babyai=5 \
      #eval.num_episodes.textworld=5

    echo "Completed evaluation for ${MODEL_ID}"
    echo ""
done

echo "All evaluations finished!"

