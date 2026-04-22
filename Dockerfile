FROM python:3.11-slim

WORKDIR /app

COPY requirement.txt .

RUN pip install --no-cache-dir -r requirement.txt

COPY . .

EXPOSE 80

CMD ["python", "application_deploy_1.py"]
