# LLaMA-Inspired-Model
This project implements a transformer-based language model inspired by Meta’s LLaMA architecture.

## Model Overview
The model is built using PyTorch and features key innovations from LLaMA, such as:

- RMSNorm: Root Mean Square Layer Normalization for improved training stability.

- Rotary Positional Embeddings (RoPE): Efficiently encodes positional information in the attention mechanism.

- SwiGLU Activation: A gated linear unit variant for the feedforward network, enhancing model expressiveness.

- Causal Masked Multi-Head Self-Attention: Ensures autoregressive (left-to-right) text generation.

- Configurable Depth and Width: Easily adjust the number of layers, attention heads, and embedding size.

## Installation
Clone the repository:
```
git clone git@github.com:milojkonikolic/LLaMA-Inspired-Model.git
```
```
cd LLaMA-Inspired-Model
```
Install required dependencies:
```
pip install -r requirements.txt
```

## Usage

To train the model, run the `train.py` script with the path to the configuration file (`config.yaml`). The configuration file defines the core model parameters, logging options, and key hyperparameters.

Training logs are displayed in the console and also recorded for TensorBoard. To launch TensorBoard, use:
```
tensorboard --logdir <path-to-logs> --port <port>
```

Since the model is autoregressive and is trained from scratch, it periodically prints the predicted tokens to the console during training.

## Dataset

Dataset that is used for training is WikiText2 - collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. Dataset can be found [here](https://huggingface.co/datasets/mindchain/wikitext2).

## Results and Evaluation
Evaluation of the model can be run using the script `eval.py`. Evaluation script prints the loss and perplexity metrics.
The input to the eval script is path to configuration file used for training (it is saved in `output_dir` specified for training). Another argument that should be provided is `step` which chooses the saved checkpoint for eval (if not specified the latest model will be used by default).

The model is trained on NVIDIA GeForce RTX 3070 GPU with 8GB of VRAM. Due to hardware constraints, the following model parameters are used:
```
batch_size: 2
vocab_size: 50257
embedding_dim: 256
context_length: 256
num_heads: 8
num_decoders: 8
```
This configuration results in a significantly smaller model compared to the original LLaMA architecture, which means the performance and capacity of the trained model are also limited.

## License
LLaMA-Inspired-Model is licensed under the MIT License.

## Acknowledgments
This project is inspired by  the original LLaMA paper that can be found [here](https://arxiv.org/pdf/2302.13971).
