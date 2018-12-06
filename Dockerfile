FROM tensorflow/tensorflow:latest-gpu

RUN mkdir /app
RUN mkdir /app/code

COPY ./.git/refs/heads/master /app/commit-id
COPY ./code/*.py /app/code/
COPY Makefile /app

RUN pip install --trusted-host pypi.python.org requests tqdm pathlib
RUN chmod a+x /app/code/fair_networks.py /app/code/random_networks.py /app/code/test_network.py /app/code/test_representations.py
ENV PATH="/app/code:${PATH}"

CMD ["python", "./fair_networks.py", "-h"]

