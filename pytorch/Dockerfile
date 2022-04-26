FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y sudo g++ make libsndfile1 --no-install-recommends

WORKDIR /code

COPY mecab-0.996-ko-0.9.2.tar.gz .
RUN tar xvfz mecab-0.996-ko-0.9.2.tar.gz && cd mecab-0.996-ko-0.9.2 && ./configure && make && make check && sudo make install

COPY mecab-ko-dic-2.1.1-20180720.tar.gz .
RUN tar zxfv mecab-ko-dic-2.1.1-20180720.tar.gz && cd mecab-ko-dic-2.1.1-20180720 && sudo ldconfig && ./configure && make && sudo make install

COPY g2pK g2pK/
RUN cd g2pK && pip install --no-cache-dir .

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
