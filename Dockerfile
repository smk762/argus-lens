FROM gpu_base

WORKDIR /app

COPY dist/*.whl /tmp/wheels/
RUN --mount=type=cache,target=/root/.cache/pip \
    set -- /tmp/wheels/*.whl && \
    python -m pip install \
        "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70" && \
    python -m pip install "certifi>=2026.10.19" "$1[server,local,openai,replicate,lineage]" && \
    rm -rf /tmp/wheels

EXPOSE 8080
CMD ["argus-lens", "serve", "--port", "8080", "--cors"]
