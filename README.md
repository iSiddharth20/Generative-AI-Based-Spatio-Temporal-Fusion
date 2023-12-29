# GenAI-Powered Spatio-Temporal Fusion for Video Super-Resolution
![GitHub Latest Release)](https://img.shields.io/github/v/release/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion?logo=github)
![License](https://img.shields.io/github/license/iSiddharth20/Spatio-Temporal-Fusion-in-Remote-Sensing)

#### Based on PyTorch, Install [Here](https://pytorch.org/get-started/locally/)

Exploring the forefront of generative AI to enhance video quality through advanced spatio-temporal fusion techniques by Upscaling and Frame-Interpolation, leveraging Auto-Encoders, LSTM Networks and Maximum Entropy Principle.

## Introduction
Developing a novel approach to video super-resolution by harnessing the potential of Auto-Encoders, LSTM Networks, and the Maximum Entropy Principle. The project aims to refine the spatial and temporal resolution of video data, unlocking new possibilities in high-resolution, high-fps, more-color-dense videos and beyond.

## Research Objective

The main goals of the project are:
- To learn temporal dependencies among spatially-sparse-temporally-dense greyscale image frames to predict and interpolate new frames, hence, increasing temporal resolution.
- To learn spatial dependencies through spatially-dense-temporally-sparse sequences that include both greyscale and corresponding RGB image frames to generate colorized versions of greyscale frames, thus, enhancing spatial resolution.

Here's a visual representation of the data transformation:
- **Current Format**: `[Grey-1] [Grey-2, RGB-2] [Grey-3] [Grey-4] ... [Grey-8, RGB-8] [Grey-9] [Grey-10]`
- **Post-Processing**: `[RGB-1] [RGB-1.5] [RGB-2] [RGB-2.5] ... [RGB-8.5] [RGB-9] [RGB-9.5] [RGB-10]`

## Resource Links

- [Code Explanation](CodeExplanation.md) 
- [Issue Tracker](https://github.com/iSiddharth20/Spatio-Temporal-Fusion-in-Remote-Sensing/issues)
- [Dataset Access](https://www.kaggle.com/datasets/isiddharth/spatio-temporal-data-of-moon-rise-in-raw-and-tif)

## Contributions Welcome!
Your interest in contributing to the project is highly respected. Aiming for collaborative excellence, your insights, code improvements, and innovative ideas are highly appreciated. Make sure to check [Contributing Guidelines](CONTRIBUTING.md) for more information on how you can become an integral part of this project.

## Acknowledgements
A heartfelt thank you to all contributors and supporters who are on this journey to break new ground in video super-resolution technology.
