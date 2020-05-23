FROM hiroshiba/hiho-deep-docker-base:pytorch1.5.0-cuda9.0

WORKDIR /app

# install requirements
RUN pip install pytest mock typing-extensions filelock matplotlib torchvision==0.3.0

# add applications
COPY pytorch_trainer /app/pytorch_trainer
COPY tests /app/tests
COPY examples /app/examples
COPY setup.py /app/setup.py

CMD bash
