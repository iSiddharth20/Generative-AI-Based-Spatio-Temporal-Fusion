# GenAI-Powered Spatio-Temporal Fusion for Video Super-Resolution
![GitHub Latest Release)](https://img.shields.io/github/v/release/iSiddharth20/Generative-AI-Based-Spatio-Temporal-Fusion?logo=github)
![License](https://img.shields.io/github/license/iSiddharth20/Spatio-Temporal-Fusion-in-Remote-Sensing)

#### Based on PyTorch, Install [Here](https://pytorch.org/get-started/locally/)

Exploring the forefront of generative AI to enhance video quality through advanced spatio-temporal fusion techniques by Upscaling and Frame-Interpolation, leveraging Auto-Encoders, LSTM Networks and Maximum Entropy Principle.

## Introduction
This project is a novel approach to enhance video resolution both spatially and temporally using generative AI techniques. By leveraging Auto-Encoders and LSTM Networks, the project aims to interpolate high-temporal-resolution grayscale images and colorize them by learning from a corresponding set of RGB images, ultimately achieving high-fidelity video super-resolution.


## Research Objective
The main goals of the project are:
- To learn temporal dependencies among spatially-sparse-temporally-dense greyscale image frames to predict and interpolate new frames, hence, increasing temporal resolution.
- To learn spatial dependencies through spatially-dense-temporally-sparse sequences that include both greyscale and corresponding RGB image frames to generate colorized versions of greyscale frames, thus, enhancing spatial resolution.


Here's a visual representation of the data transformation:
- **Current Format**: ` [Grey-1] [Grey-2, RGB-2] [Grey-3] [Grey-4] ... [Grey-8, RGB-8] [Grey-9] [Grey-10]`
- **Post-Processing**: `[RGB-1] [RGB-1.5] [RGB-2] [RGB-2.5] ... [RGB-8.5] [RGB-9] [RGB-9.5] [RGB-10]`

## Resource Links

- [Code Explanation](CodeExplanation.md) 
- [Issue Tracker](https://github.com/iSiddharth20/Spatio-Temporal-Fusion-in-Remote-Sensing/issues)

## Contributions Welcome!
Your interest in contributing to the project is highly respected. Aiming for collaborative excellence, your insights, code improvements, and innovative ideas are highly appreciated. Make sure to check [Contributing Guidelines](CONTRIBUTING.md) for more information on how you can become an integral part of this project.

## Acknowledgements
A heartfelt thank you to all contributors and supporters who are on this journey to break new ground in video super-resolution technology.
