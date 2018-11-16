FROM tensorflow/tensorflow:latest-gpu
RUN mkdir /app
RUN mkdir /app/data
RUN mkdir /app/code
COPY ./code/*.py /app/code/
COPY ./data/* /app/data/
WORKDIR /app
RUN pip install --trusted-host pypi.python.org requests tqdm pathlib
CMD ["python", "./fair_networks.py", "-h"]

