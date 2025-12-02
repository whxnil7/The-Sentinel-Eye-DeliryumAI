FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		libgl1 \
		libglib2.0-0 && \
	rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py", "--no-gui", "--output-video", "data/output/output_docker.mp4"]
