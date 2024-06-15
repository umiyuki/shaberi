python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-7B-Instruct \
    --chat-template Qwen2-7B-Instruct.jinja \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096