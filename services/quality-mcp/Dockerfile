FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --system cortex \
    && useradd --system --gid cortex --home-dir /app cortex

COPY services/common/cortex_common ./cortex_common
COPY services/quality-mcp/quality_mcp ./quality_mcp
COPY scripts ./scripts

USER cortex
EXPOSE 8100

CMD ["python", "-m", "quality_mcp.app"]
