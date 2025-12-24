# Gunakan Python versi kecil
FROM python:3.10-slim

# Set folder kerja
WORKDIR /app

# Copy file requirements (conda.yaml kita anggap list manual dulu biar simpel di docker)
RUN pip install mlflow pandas scikit-learn

# Copy semua file ke dalam image
COPY . /app

# Perintah saat container jalan
CMD ["python", "modelling.py"]