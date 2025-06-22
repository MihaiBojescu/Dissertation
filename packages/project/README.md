# Music generation using Diffusion Models: Project

## Description

This package hosts the project for my dissertation.

## Table of contents

- [Music generation using Diffusion Models: Project](#music-generation-using-diffusion-models-project)
  - [Description](#description)
  - [Table of contents](#table-of-contents)
  - [Content](#content)
  - [Setup](#setup)
    - [Dependencies](#dependencies)
    - [Training prerequisites](#training-prerequisites)
  - [Running](#running)
    - [Training](#training)
    - [Inferring](#inferring)

## Content

| Path                                         | Description                                                                                                       |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| [`./data/input`](./data/input)               | The audio files                                                                                                   |
| [`./data/output`](./data/output)             |                                                                                                                   |
| [`./data/spectrogram`](./data/spectrogram)   | The spectrograms, produced after the audio files are preprocessed                                                 |
| [`./data/weights`](./data/weights)           | The model weights, produced after training                                                                        |
| [`./scripts`](./scripts)                     | Model scripts, like inference, preprocessing and training                                                         |
| [`./src`](./src)                             | The source files for the project                                                                                  |
| [`./src/dataset`](./src/dataset)             | The source files for dataset processing                                                                           |
| [`./src/interfaces`](./src/interfaces)       | The source files for interface                                                                                    |
| [`./src/model`](./src/model)                 | The source files for the PyTorch model                                                                            |
| [`./src/preprocessing`](./src/preprocessing) | The source files used for preprocessing the data                                                                  |
| [`./src/training`](./src/training)           | The source files used for training the model                                                                      |
| [`./src/utils`](./src/utils)                 | The source files containing utility functions, like noising, collation, function to choose the appropriate device |
| [`./.vscode`](./.vscode)                     | VSCode configuration files                                                                                        |
| [`.venv`](./venv)                            | Files supporting the project                                                                                      |

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
