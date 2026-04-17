FROM gpu_base

WORKDIR /app

COPY dist/*.whl /tmp/wheels/
RUN --mount=type=cache,target=/root/.cache/pip \
    set -- /tmp/wheels/*.whl && \
    pip install --upgrade pip && \
    pip install "$1[server,local,openai,replicate]" && \
    rm -rf /tmp/wheels

EXPOSE 8080
CMD ["argus-lens", "serve", "--port", "8080", "--cors"]
