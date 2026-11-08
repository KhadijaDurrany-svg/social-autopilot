# Minimal, offline-by-default image for the slotwise CLI.
FROM python:3.12-slim AS base

# Do not write .pyc files; unbuffered stdout for predictable logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install the package first (leveraging layer caching for deps).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

# Create and switch to a non-root user.
RUN useradd --create-home --uid 10001 slotwise
USER slotwise

# Default command runs a small benchmark without writing files.
ENTRYPOINT ["slotwise"]
CMD ["bench", "--scenarios", "50", "--no-write"]
