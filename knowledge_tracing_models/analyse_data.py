import numpy as np
import pandas as pd
import sys


def analyse_dataset(data):
    user_ids = pd.unique(data["user_id"])

    df = data.groupby(['user_id']).size()
    print(df)
    print(df.value_counts().to_string())
    max_iter = df.max()
    print(max_iter)
    content = []
    val_counts = df.value_counts()
    for x in val_counts:
        content.append([val_counts])


if __name__ == '__main__':
    dataset = "../data/data.csv"
    analyse_dataset(pd.read_csv(dataset))