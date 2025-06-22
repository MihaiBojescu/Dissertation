# Music generation using Diffusion Models: Documentation

## Description

This package hosts the documentation for my dissertation.

## Table of contents

- [Music generation using Diffusion Models: Documentation](#music-generation-using-diffusion-models-documentation)
  - [Description](#description)
  - [Table of contents](#table-of-contents)
  - [Content](#content)
  - [Setup](#setup)
  - [Results](#results)

## Content

| Path                                         | Description                                                                                                                                                |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`./src`](./src)                             | The source files for the paper                                                                                                                             |
| [`./res`](./res)                             | The resource files for the paper, including entities like images                                                                                           |
| [`./out/paper`](./out/paper)                 | The built paper file                                                                                                                                       |
| [`./out/presentations`](./out/presentations) | Extra presentation files, manually made                                                                                                                    |
| [`./build`](./build)                         | The folder containing the built paper file, and its auxiliary files. The `main.pdf` file will be moved after it is built into [`./out/paper`](./out/paper) |
| [`./.vscode`](./.vscode)                     | VSCode configuration files                                                                                                                                 |

## Setup

Building the paper requires TeX Live to be installed. On Arch-based distributions, use:

```sh
$ sudo pacman -Syu texlive texlive-lang texlive-binextra
```

## Results

Outputs are put into `./build/main.pdf`. If the outputs passes the inspection, they will be moved using

```sh
$ mv ./build/main.pdf ./out/paper/main.pdf
```

to the output folder.
