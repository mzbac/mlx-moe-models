# MLX-MOE Models Package

The `mlx-moe` package offers custom MLX Mixture of Experts (MoE) models, streamlining the process of leveraging sophisticated MoE models for text generation, particularly in conjunction with the `mlx-lm` package. This tool is designed for developers and researchers who require advanced text generation capabilities, providing an easy pathway to utilize and integrate MoE models into their projects.

## Features

- **Custom MoE Model Support**: Easily load and utilize your custom MLX MoE models for text generation.
- **Easy Installation**: Get started quickly with a simple pip installation command.

## Installation

To install `mlx-moe`, simply run the following command in your terminal:

```shell
pip install mlx-moe
```

## Usage

The `mlx-moe` package is designed to be used in conjunction with the mlx-lm package for generating text. After installing mlx-moe, you can load your custom MoE models as follows:
```
from mlx_moe.load import load_moe

model_path = "path_to_your_model"
model = load_moe(model_path)
```