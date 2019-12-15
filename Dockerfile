FROM hiroshiba/hiho-deep-docker-base:pytorch1.1-cuda9.0

WORKDIR /app

# install requirements
RUN pip install pytest mock typing-extensions filelock matplotlib torchvision==0.3.0

# add applications
COPY chainer /app/chainer
COPY tests /app/tests
COPY examples /app/examples

CMD bash
