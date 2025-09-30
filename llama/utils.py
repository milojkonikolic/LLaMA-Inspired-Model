import os
import torch


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
