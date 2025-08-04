
#!/bin/bash

set -e

# Default values if not set
: ${MODEL_PATH:="/model"}
: ${VLLM_PORT:=11434}
: ${API_ADAPTER_HOST:="0.0.0.0"}
: ${API_ADAPTER_PORT:=8003}
: ${SERVICED_MODEL_NAME:="llama4"}
: ${TOKENIZER_PATH:="${MODEL_PATH}"}
: ${TENSOR_PARALLEL_SIZE:=1}
: ${VLLM_STARTUP_DELAY:=6000}
: ${MAX_MODEL_LEN:=32768}

python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --host 0.0.0.0 \
  --port "${VLLM_PORT}" \
  --served-model-name "${SERVICED_MODEL_NAME}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 128 \
  --trust-remote-code &

# Wait for vLLM to start
sleep ${VLLM_STARTUP_DELAY}

export OPENAI_BASE_URL_INTERNAL="http://localhost:${VLLM_PORT}"
export OPENAI_BASE_URL="http://localhost:${API_ADAPTER_PORT}"
export API_ADAPTER_HOST="${API_ADAPTER_HOST}"
export API_ADAPTER_PORT="${API_ADAPTER_PORT}"
export ENABLE_MCP_TOOLS=false

uvicorn open_responses_server.server_entrypoint:app --host "${API_ADAPTER_HOST}" --port "${API_ADAPTER_PORT}" 