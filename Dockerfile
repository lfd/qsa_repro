FROM ubuntu:20.04

LABEL authors="Maja Franz <maja.franz@othr.de>, Manuel Schoenberger <manuel.schoenberger@othr.de>"

ENV DEBIAN_FRONTEND noninteractive
ENV LANG="C.UTF-8"
ENV LC_ALL="C.UTF-8"

# Install required packages
RUN apt-get update && apt-get install -y \
        python3 \
        python3-pip \
        texlive-latex-base \
        texlive-science \
        texlive-fonts-recommended \
        texlive-publishers \
        texlive-bibtex-extra \
        biber

# Add user
RUN useradd -m -G sudo -s /bin/bash repro && echo "repro:repro" | chpasswd
RUN usermod -a -G staff repro
USER repro

# Add artifacts (from host) to home directory
ADD --chown=repro:repro . /home/repro/qsa-repro

WORKDIR /home/repro/qsa-repro

# install python packages
ENV PATH $PATH:/home/repro/.local/bin
RUN pip3 install -r requirements.txt

# Run all RL and MQO experiments and generate paper when container is started
ENTRYPOINT ["./scripts/run.sh"]
CMD ["bash"]
