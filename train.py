import os
import argparse
import yaml
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from llama.model import LLaMA
from llama.dataset import get_data_loader, get_tokenizer
from llama.utils import save_model
from utils import get_logger, get_gpu, log_to_tb, log_to_console


def save_config(config):
    os.makedirs(config["output_dir"], exist_ok=True)
    config_dst = os.path.join(config["output_dir"], os.path.basename(args.config))
    with open(config_dst, "w") as f:
        yaml.safe_dump(config, f)


def train(config, tb_writer, logger, device):
    model = LLaMA(config["vocab_size"], config["embedding_dim"], config["num_heads"], 
                  config["context_length"], config["num_decoders"], config["batch_size"])
    model.to(device)

    tokenizer = get_tokenizer()
    data_loader = get_data_loader(config["train_data"], tokenizer, 
                                  config["context_length"], config["batch_size"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(config["epochs"]):
        for batch in tqdm(data_loader):
            inputs, target = batch
            inputs = inputs.to(device)
            target = target.to(device)
            outputs = model(inputs)

            loss = criterion(outputs.view(-1, outputs.size(-1)), target.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_to_tb(tb_writer, loss.item(), optimizer.param_groups[0]["lr"], global_step)
            log_to_console(config["log_interval"], global_step, epoch, 
                           loss, outputs, tokenizer, logger)
            global_step += 1
            
            if global_step % 10000 == 0:
                save_model(model, config["output_dir"], global_step, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger = get_logger()
    device = get_gpu(config["gpu_id"], logger)
    tb_writer = SummaryWriter(log_dir=os.path.join(config["output_dir"], "logs"))

    save_config(config)    

    train(config, tb_writer, logger, device)
