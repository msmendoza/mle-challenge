FROM python:3.13-slim-bullseye

# copy local code to the container image.
WORKDIR /app
COPY . .

# install production dependencies.
RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["fastapi", "run", "challenge/api.py", "--host", "0.0.0.0", "--port", "8080"]
