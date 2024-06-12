python -m vllm.entrypoints.openai.api_server \
    --model HODACHI/glm-4-9b-chat-FT-ja-v0.3 \
    --chat-template /home/umiyuki/shaberi/shaberi/chat_templates/glm-4-9b-chat-FT-ja-v0.3.jinja \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096