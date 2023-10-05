import json
import torch
from torch.utils.data import DataLoader

MAX_LEN = 16
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32

class CustomDataset:
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.input =  dataframe["input"]
        self.targets =  self.data.target_list
        self.max_len = max_len

    def __getitem__(self, index):
        input = str(self.input[index])
        input = " ".join(input.split())

        inputs = self.tokenizer.encode_plus(
            input,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = input["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[index], dtype=torch.float),
        }


def df_loader(df, batch_size, tokenizer):
    custom = CustomDataset(df, tokenizer, MAX_LEN)
    test_params = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 0
        }
    return DataLoader(custom, **test_params)


def get_categories():
    with open("observation_categories.json", "r") as f:
        categories = json.load(f)["categories"]
    return categories


def df_loader(df, batch_size, tokenizer):
    custom = CustomDataset(df, tokenizer, MAX_LEN)
    test_params = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 0
        }
    return DataLoader(custom, **test_params)


def feature_prep(dataset):
    df = dataset[["whatdidyousee", "category2"]]

    df = df.loc[df["category2"].notnull()]
    df = df.reset_index(drop=True)

    df["category"] = df["category2"].str.strip()
    category_list = get_categories()

    for category in category_list:
        df.loc[df["category"] == category, category] = 1
        df[category] = df[category].fillna(0)

    df["input"] = df["whatdidyousee"].astype(str)
    df["target_list"] = df[category_list].astype(bool).values.tolist()

    return df[["input", "category", "target_list"]]

def split_sample(data, train_size):
    train_dataset = data.sample(frac=train_size)
    valid_dataset = data.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print(f"FULL Dataset: {data.shape}")
    print(f"TRAIN Dataset: {train_dataset.shape}")
    print(f"TEST Dataset: {valid_dataset.shape}")

    training_loader = df_loader(train_dataset, TRAIN_BATCH_SIZE)
    validation_loader = df_loader(valid_dataset, VALID_BATCH_SIZE)
    return training_loader, validation_loader