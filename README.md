# Music generation using Diffusion Models

## Description

This repository hosts my Artificial Intelligence and Optimisation Master's Thesis: A way to generate music using diffusion models.

## Table of contents

- [Music generation using Diffusion Models](#music-generation-using-diffusion-models)
  - [Description](#description)
  - [Table of contents](#table-of-contents)
  - [Setup](#setup)
    - [Dependencies](#dependencies)
    - [Training prerequisites](#training-prerequisites)
  - [Running](#running)
    - [Training](#training)
    - [Inferring](#inferring)

## Setup

The setup process aims to be simple and straight-forward. Please follow the steps bellow.

### Dependencies

Follow these steps:

1. Install Python >= 3.12
2. Install [Poetry](https://python-poetry.org/)
3. Run `$ poetry install`
4. Run `$ eval $(poetry env activate)`

### Training prerequisites

The project requires input audio data to be placed in [./data/input] and needs to be in one of the following formats:

- `flac`
- `m4a`
- `mp4`
- `wma`
- `wav`
- `mp3`
- `aiff`

## Running

### Training

Having setup the prerequisites and the dependencies, the input data can be preprocessed. To perform this, use:

```sh
$ ./scripts/preprocess
```

After which, the model can be trained using:

```sh
$ ./scripts/train
```

After training, the model weights will be placed in `./data/weights/`, with the format `<iso-8601 timestamp>.pt`.

### Inferring

TBD
