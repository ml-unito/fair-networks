FROM tensorflow/tensorflow:latest-gpu

RUN mkdir /app
RUN mkdir /app/bin
RUN mkdir /app/packages

COPY ./.commit-id /app
COPY ./packages /app/packages/
COPY ./bin/ /app/bin/
COPY ./docker_requirements.txt /app
COPY Makefile /app

RUN pip install -r /app/docker_requirements.txt
ENV PATH="/app/bin:${PATH}"
ENV PYTHONPATH="/app/packages:${PYTHONPATH}"

CMD ["fair_networks", "-h"]

