FROM python:3.8
WORKDIR /app
RUN pip install flask scikit-learn pandas joblib
COPY . .
CMD ["python", "app.py"]