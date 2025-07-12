FROM rocm/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && \
    apt autoremove -y && \
    apt install -y gnupg wget curl zsh git && \
    rm -rf /var/lib/apt/lists/*
RUN chsh -s /bin/zsh jenkins

USER jenkins

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN /opt/conda/envs/py_3.12/bin/python3 -m pip install --user pipx && \
    /opt/conda/envs/py_3.12/bin/python3 -m pipx ensurepath && \
    /opt/conda/envs/py_3.12/bin/python3 -m pipx install poetry
