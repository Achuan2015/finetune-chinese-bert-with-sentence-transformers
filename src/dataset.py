from sentence_transformers import InputExample

class Dataset:

    def read(self, data, return_pt=False):
        sentence1 = data['sent1'].tolist()
        sentence2 = data['sent2'].tolist()
        labels = data['label'].tolist()
        if return_pt:
            dataloader = []
            for s1, s2, l in zip(sentence1, sentence2, labels):
                dataloader.append(InputExample(texts=[s1, s2], label=l))
            return dataloader
        return sentence1, sentence2, labels


if __name__ == "__main__":
    import pandas as pd
    from sklearn import model_selection
    dataset = Dataset()
    dfs = pd.read_csv("../data/debug_train.csv", sep="\t", names=['idx', 'sent1', 'sent2', 'label']) 
    dfs["label"] = pd.to_numeric(dfs["label"])

    df_train, df_valid = model_selection.train_test_split(
            dfs,
            test_size=0.1,
            random_state=42,
            stratify=dfs.label.values
    )
    datasets = dataset.read(df_train)
