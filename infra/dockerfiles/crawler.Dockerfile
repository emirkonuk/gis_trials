FROM mcr.microsoft.com/playwright:v1.49.0-jammy

# add pip
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /project

# python deps for crawler
COPY src/crawler/requirements.txt /project/src/crawler/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /project/src/crawler/requirements.txt

# ensure browsers are present for Python API
ENV PYTHONUNBUFFERED=1 PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN python3 -m playwright install --with-deps firefox

CMD ["bash","-lc","python3 -c 'print(\"crawler ready\")'"]

