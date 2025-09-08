FROM python:3.12-slim

RUN pip3 install --upgrade pip
WORKDIR /app

COPY requirements-app.txt requirements-app.txt
RUN pip3 install --no-cache-dir -r requirements-app.txt

COPY core/ core/
COPY app/ app/
COPY config.json config.json

# Init Gunicorn https://flask.palletsprojects.com/en/3.0.x/deploying/gunicorn/
# EXPOSE 8000  # not needed with k8s as we will specify ports there.
# max-requests: how many requests before worker restarts
# max-requests-jitter: adds random number to max-requests with ceiling at max-requests-jitter to prevent all workers from restarting at once
# --timeout causes gunicorn to restart worker if it hasn't done work in X seconds. Default is 30s. Lower timeouts are better for high request volumes
# CMD ["gunicorn", "-b", "0.0.0.0:8000", "--workers=1", "--max-requests=1200", "--max-requests-jitter=50", "--log-level", "info", "app.app:app"]
CMD ["gunicorn", "-b", "0.0.0.0:8000", "--workers=1", "--max-requests=1200", "--max-requests-jitter=50", "--capture-output", "--log-file", "-", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "app.app:app"]
