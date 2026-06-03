#!/bin/bash
# 两人跑得快 16-hand 变种 AI 推理服务 —— Docker 启停脚本
# 用法:
#   ./start.sh                                # 默认 PORT=7788, MODEL=checkpoints/194M.pt
#   PORT=8000 MODEL=checkpoints/best.pt ./start.sh
set -e

IMAGE_NAME="run_fast_2p_16"
CONTAINER_NAME="run_fast_2p_16_service"
PORT="${PORT:-7788}"
MODEL="${MODEL:-checkpoints/194M.pt}"

# 校验模型存在
if [ ! -f "$MODEL" ]; then
    echo "[start.sh] ERROR: model file not found: $MODEL"
    echo "[start.sh] checkpoints/ 下的可用模型:"
    ls -1 checkpoints/*.pt 2>/dev/null || echo "  (无)"
    exit 1
fi

# Build
echo "[start.sh] Building image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .

# Remove existing container if present
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[start.sh] Removing existing container: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME"
fi

# Run
echo "[start.sh] Starting container: $CONTAINER_NAME on port $PORT (model=$MODEL)"
docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -p "${PORT}:7788" \
    -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
    -e MODEL="$MODEL" \
    "$IMAGE_NAME" \
    python src/service/service.py --model "$MODEL" --port 7788

# Wait for service to be ready (max 30s)
echo -n "[start.sh] Waiting for health_check"
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${PORT}/health_check" > /dev/null 2>&1; then
        echo " OK"
        curl -s "http://localhost:${PORT}/health_check" && echo
        echo "[start.sh] Service ready at: http://localhost:${PORT}"
        echo "[start.sh] API doc:           src/service/API.md"
        echo "[start.sh] Logs:              docker logs -f $CONTAINER_NAME"
        echo "[start.sh] Stop:              docker rm -f $CONTAINER_NAME"
        exit 0
    fi
    echo -n "."
    sleep 1
done
echo " TIMEOUT"
echo "[start.sh] Service did not respond within 30s. Container logs:"
docker logs --tail 50 "$CONTAINER_NAME"
exit 1
