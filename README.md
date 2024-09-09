## Neural Decoder for 3D Toric Codes
![4D Torus](torus.gif)

This repository contains code for experiments using a Neural Decoder that works on 3D Toric codes.
The corresponding paper can be found at: http://arxiv.org/abs/2409.04300 .

- `models` contains various decoder models and sub-packages with additional components.

  - `auxiliar_components` contains various components used in the NNs or decoding procedure.
  - `pooling_layers` contains adaptations of the GlobalAveragePooling layer.
  - `residual_block` contains different types of residual block implementations.
  
- `src` is a collection of classes and functions used in the experimentation.

  - `data_analysis` is a collection of functions for data analysis.
  - `evaluation_metrics` collects various evaluation methods for the decoding.

- `config` contains the configuration files used for experiments.

  - `default` contains default parameters for the experiments and API details to W&B.
  - `net` contains the specific parameters for the network architecture.
