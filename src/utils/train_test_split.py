def train_test_split(data, train_size_split:int):
    train_size = int(len(data)*train_size_split)
    train = data.iloc[:train_size, :]
    test = data.iloc[train_size:, :]

    return train, test