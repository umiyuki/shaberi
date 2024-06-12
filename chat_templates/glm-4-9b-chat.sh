python -m vllm.entrypoints.openai.api_server \
    --model THUDM/glm-4-9b-chat \
    --chat-template glm-4-9b-chat.jinja \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096