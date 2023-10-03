def feature_prep(dataset):
    df = dataset[["whatdidyousee", "category2"]]

    df = df.loc[df["category2"].notnull()]
    df = df.reset_index(drop=True)

    df["category"] = df["category2"].str.strip()
    category_list = get_categories()

    
    raise NotImplementedError()

def split_sample():
    raise NotImplementedError()