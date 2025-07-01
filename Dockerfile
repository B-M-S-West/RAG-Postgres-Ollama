FROM python:3.12-slim-bookworm

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
COPY . /app

# Install the application dependencies.
WORKDIR /app
RUN uv sync --frozen --no-cache

# Create logs directory
RUN mkdir -p logs

# Entrypoint script to run both backend and frontend (This will need some refining)
RUN echo '#!/bin/bash\n\
uv run fastapi dev --host 0.0.0.0 --port 8001 &\n\
sleep 5\n\
streamlit run app/frontend.py --server.address 0.0.0.0 --server.port 8501\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]

# Build the image
#docker build -t rag-app .

# Run the container
#docker run -p 8001:8001 -p 8501:8501 rag-app