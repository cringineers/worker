FROM docker.io/python:3.10-slim
ENV PATH /app/venv/bin:$PATH
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install git -y &&\
    python -m venv venv && pip install wheel --no-cache-dir &&\
    pip install --no-cache-dir -r requirements.txt &&\
    rm -rf /var/lib/apt/lists/*
COPY . /app
ENTRYPOINT ["python", "run.py"]
