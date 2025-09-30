import os
import argparse
import yaml
import torch
from tqdm import tqdm

from llama.model import LLaMA
from llama.dataset import get_data_loader, get_tokenizer
from utils import get_logger, get_gpu


def evaluate(config, logger, device, step=None):
    
    model = LLaMA(config["vocab_size"], config["embedding_dim"], config["num_heads"], 
                  config["context_length"], config["num_decoders"], config["batch_size"])
    model.to(device)

    checkpoint_dir = os.path.join(config["output_dir"], "checkpoints")
    # Load the latest checkpoint if step is not specified
    if step is None:
        step = max([int(f.split('_')[-1].split('.')[0]) for f in os.listdir(checkpoint_dir) if f.startswith("llama-inspired_")])
    model_path = os.path.join(config["output_dir"], "checkpoints", f"llama-inspired_{step}.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"Loaded model checkpoint from {model_path}")

    model.eval()
    
    tokenizer = get_tokenizer()
    data_loader = get_data_loader(config["eval_data"], tokenizer, 
                                  config["context_length"], config["batch_size"])

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs, target = batch
            inputs = inputs.to(device)
            target = target.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target.view(-1))
            total_loss += loss.item() * inputs.size(0)
            total_tokens += inputs.size(0) * inputs.size(1)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    logger.info(f"Evaluation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the saved config file.")
    parser.add_argument("--step", type=int, default=None,
                        help="Step number of the checkpoint to load.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger = get_logger()
    device = get_gpu(config["gpu_id"], logger)

    evaluate(config, logger, device, args.step)
