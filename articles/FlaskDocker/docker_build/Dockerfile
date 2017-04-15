FROM python:3

COPY . /app

WORKDIR /app

RUN pip install -r /app/requirements.txt

EXPOSE 5000

ENTRYPOINT ["python"]

CMD ["api.py"]
