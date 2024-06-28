## Neural Decoder for 3D Toric Codes
![4D Torus](experimentation/torus.gif)

This repository contains code for experiments using a Neural Decoder that works on 3D Toric codes.

- `models` contains various decoder models and sub-packages with additional components.

  - `auxiliar_components` contains various components used in the NNs or decoding procedure.
  - `pooling_layers` contains adaptations of the GlobalAveragePooling layer.
  - `residual_block` contains different types of residual block implementations.
  
- `experimentation` is a directory containing experiment, testing and analysis related code.
- `src` is a collection of classes and functions used in the experimentation.

  - `data_analysis` is a collection of functions for data analysis.
  - `evaluation_metrics` collects various evaluation methods for the decoding.


