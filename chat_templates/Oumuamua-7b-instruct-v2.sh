python -m vllm.entrypoints.openai.api_server \
    --model nitky/Oumuamua-7b-instruct-v2 \
    --chat-template /home/umiyuki/shaberi/shaberi/chat_templates/Oumuamua-7b-instruct-v2.jinja \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096