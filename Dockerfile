FROM python:3.10-slim

LABEL project="run_fast_2p_16_v0"
LABEL description="两人跑得快 16-hand 变种 AI 推理服务"

WORKDIR /app

# 先装依赖（充分利用 layer cache）
COPY src/service/requirements.txt .
# torch CPU-only 版本可显著减小镜像体积（推理 CPU 足够）
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt

# 拷贝项目源码
COPY . .

EXPOSE 7788

# checkpoints/ 由 start.sh 通过 -v 挂载，不在 image 里
CMD ["python", "src/service/service.py", "--model", "checkpoints/194M.pt", "--port", "7788"]
