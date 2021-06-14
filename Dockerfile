FROM python:3.7

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY . .

EXPOSE 8080
CMD ["opyrator", "launch-api", "server:get_question_contents"]