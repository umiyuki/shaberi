python -m vllm.entrypoints.openai.api_server \
    --model umiyuki/Llama-3-Umievo-itr014-Shizuko-8b \
    --chat-template Llama-3.jinja \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096