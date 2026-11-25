# The image the agent and every code runner run in: Python 3.11, NumPy, SciPy, msgpack and the
# inkling_robo SDK. No simulator, no pip, no setuid binaries, non-root.
# Build with docker/agent/build.sh (the build context is docker/agent/ and sdk/inkling_robo/).
#
# Nothing here is relied on for security: ContainerSandbox passes the user, entrypoint, command
# and environment explicitly on every `docker run`.

# python:3.11-slim-bookworm (Python 3.11.18), multi-arch index digest
FROM python:3.11-slim-bookworm@sha256:ef56ae98435cc40d8a27020b0b32a9941515acd5840f0fba34858bb34dd362d1

ARG SITE=/usr/local/lib/python3.11/site-packages

COPY docker/agent/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-deps --only-binary=:all: --require-hashes \
        -r /tmp/requirements.txt \
    && pip uninstall -y -q pip setuptools wheel \
    && rm -rf /tmp/requirements.txt /root/.cache /usr/local/lib/python3.11/ensurepip \
    && find $SITE/numpy $SITE/scipy -depth -type d -name tests -exec rm -rf {} + \
    && find / -xdev -type f -perm /6000 -exec chmod a-s {} +

COPY sdk/inkling_robo $SITE/inkling_robo
RUN python -m compileall -q --invalidation-mode unchecked-hash $SITE/inkling_robo

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp LANG=C.UTF-8 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
USER 65534:65534
WORKDIR /tmp
CMD ["python", "-m", "inkling_robo.run", "/agent"]
