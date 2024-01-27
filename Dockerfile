FROM python:3.8

# Establecer un directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos al contenedor
COPY modelo_lineal.py /app/modelo_lineal.py
COPY modelo_regresion_aleatoria.py /app/modelo_aleatoria.py
COPY modelo_coeficiente_determinacion.py /app/modelo_coeficiente_determinacion.py

# Copiar el conjunto de datos al contenedor
COPY sensor-data.csv /app/sensor-data.csv

# Copiar el archivo de requisitos al contenedor
COPY requirements.txt /app/requirements.txt

# Instalar los requisitos
RUN pip install -r /app/requirements.txt

# Ejecutar el script de inicio cuando se inicie el contenedor
CMD ["python", "/app/modelo_lineal.py"]

