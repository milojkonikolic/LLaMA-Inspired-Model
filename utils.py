import os
import logging
import torch


def get_logger():
    """
    Get a logger instance.
    """
    logger = logging.getLogger("LLaMA-Inspired")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s]-[%(filename)s]: %(message)s ")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def get_gpu(gpu, logger=None):
    """
    Get the torch device for GPU or CPU.
    Args:
        gpu (str): Specify the device ("cpu" or "gpu").
        logger (logging.Logger, optional): Logger instance for logging. Default is None.
    Returns:
        torch.device: Torch device for computation.
    """
    if gpu == "cpu":
        if logger:
            logger.info("Selected CPU instead of GPU. Consider using GPU for better performance")
        return torch.device("cpu")
    else:
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid gpu {gpu} requested"
        if logger:
            logger.info(f"Selected device: GPU{gpu}")
        return torch.device(f"cuda:{gpu}")


def save_model(model, output_dir, step, logger):
    """
    Saves the state dictionary of a model to a file in the specified directory.
    Args:
        model (torch.nn.Module): Model to save weights for.
        output_dir (str): Directory path to save the model.
        step (int): Step number.
        logger (logging.Logger): Logger instance for logging.
    Returns:
        None
    """
    model_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f"llama-inspired_{step}.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved {model_path}")


def log_to_tb(tb_writer, loss, learning_rate, global_step):
    tb_writer.add_scalar("Loss", loss, global_step)
    tb_writer.add_scalar("Learning Rate", learning_rate, global_step)


def log_to_console(log_interval, global_step, epoch, loss, outputs, tokenizer, logger):
    if global_step % log_interval == 0:
        logger.info(f"Epoch: {epoch}, Batch: {global_step // (epoch + 1)}, " \
                    f"Loss: {loss.item()}")

    if global_step % (5 * log_interval) < 50:
        predicted_token_ids = torch.argmax(outputs, dim=-1)
        predicted_text = tokenizer.decode(predicted_token_ids[0], 
                                          skip_special_tokens=True)
        logger.info(f"Predicted text: {predicted_text}")

