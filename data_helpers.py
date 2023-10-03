def get_categories():
    raise NotImplementedError()


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

    return 

def split_sample():
    raise NotImplementedError()