FROM ubuntu:latest
LABEL maintainer="VinayDadi"
ENV PATH="/root/miniconda3/bin:${PATH}"
COPY "deploy/conda/mle-dev-mlflow.yml" .
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda env create -f mle-dev-mlflow.yml\
    && conda init bash\
    && conda clean --all --yes
RUN echo "conda activate mle-dev-mlflow" >> ~/.bashrc
COPY dist .
RUN . ~/.bashrc && pip install Housing_price_vinay-0.0.1-py3-none-any.whl
ENTRYPOINT ["/bin/bash", "-c", "source ~/.bashrc; exec /bin/bash"]
CMD ["python"]
