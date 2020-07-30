FROM jupyter/base-notebook:latest

ENV PYTHONUNBUFFERED 1

USER root

RUN apt-get update && apt-get install -y \
    dpkg-dev \
    gcc \
    git \
    curl \
    gnupg

USER jovyan

RUN conda install panel=0.9.7 bokeh=2.1.1 tornado param colorcet holoviews=1.13.3 hvplot=0.6.0 pandas datashader

WORKDIR /home/jovyan/work

CMD ["jupyter", "notebook", "--port=8888","--no-browser", "--ip=0.0.0.0"]
