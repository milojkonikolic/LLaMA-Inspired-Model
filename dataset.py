from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer


class LLaMADataset(Dataset):
    def __init__(self, text_data, tokenizer, context_len):
        # TODO: Instead of loading entire dataset into memory, load it in chunks
        self.tokenized_text = tokenizer(text_data, return_tensors='pt')['input_ids'].squeeze(0)
        self.context_len = context_len

    def __len__(self):
        return len(self.tokenized_text)

    def __getitem__(self, idx):
        if idx >= len(self.tokenized_text) - self.context_len - 1:
            idx = len(self.tokenized_text) - self.context_len - 2
        inputs = self.tokenized_text[idx:idx+self.context_len]
        target = self.tokenized_text[idx+1:idx+self.context_len+1]
        return inputs, target


def get_data_loader(train_data_path, tokenizer, context_length, batch_size):
    with open(train_data_path, 'r') as f:
        train_text = f.read()
    text_dataset = LLaMADataset(train_text, tokenizer, context_length)
    data_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def get_tokenizer():
    # TODO: Use LLaMA tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
