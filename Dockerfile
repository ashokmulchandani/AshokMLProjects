FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirement.txt

EXPOSE 8080

CMD ["python", "application_deploy_1.py"]
