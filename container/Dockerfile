# Build an image that can do training and inference in SageMaker with GPU access to tensorflow
# This is a Python 3.6 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM tensorflow/tensorflow:latest-gpu

LABEL maintainer="hani.mousli@gmail.com"

#tmux and vim are used to debug inside the container
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
         tmux \
         vim \
    && rm -rf /var/lib/apt/lists/*
# Here we get all python packages.
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm -rf /root/.cache

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY bert_sentiment /opt/program
WORKDIR /opt/program

