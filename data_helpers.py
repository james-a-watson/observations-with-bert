import json

def get_categories():
    with open("observation_categories.json", "r") as f:
        categories = json.load(f)["categories"]
    return categories


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

def split_sample():
    raise NotImplementedError()