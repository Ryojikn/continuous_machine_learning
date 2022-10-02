FROM ubuntu:latest

WORKDIR /app

COPY . .

RUN apt-get -y update
RUN apt-get install -y python3 pip
RUN pip3 install -r src/requirements.txt

CMD ["kedro", "run", "--pipeline"]