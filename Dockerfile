FROM python:3.11-slim

ENV PYTHONUNBUFFERED 1
ENV WORK_DIR /app
RUN mkdir -p ${WORK_DIR}
WORKDIR ${WORK_DIR}

ADD . ${WORK_DIR}/
RUN pip install --break-system-packages -r requirements.txt
RUN pip install --break-system-packages -U .

# Expose the port for Flask
EXPOSE 5000

# Entrypoint command to start the app
CMD ["python3", "main.py"]
