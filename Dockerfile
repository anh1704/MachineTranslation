# Sử dụng Python 3.8 hoặc 3.9 thay vì 3.7
FROM python:3.8

EXPOSE 8084

# Upgrade pip 
RUN pip install -U pip

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app

RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . /app

# Run the app with streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
