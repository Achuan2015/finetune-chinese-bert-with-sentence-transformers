import pandas as pd
from sklearn import model_selection
import torch.nn as nn

from sentence_transformers import evaluation
from sentence_transformers import models
from sentence_transformers import SentenceTransformer
from sentence_transformers import losses
from torch.utils.data import DataLoader

import config
import dataset
import engine


def run():
    train_file = config.TRAINING_FILE
    train_batch = config.TRAIN_BATCH_SIZE
    vaild_batch = config.VALID_BATCH_SIZE
    model_path = config.BERT_PATH
    max_length = config.MAX_LEN 
    dfs = pd.read_csv(train_file, sep="\t", names=['idx', 'sent1', 'sent2', 'label']) 
    dfs['label'] = pd.to_numeric(dfs["label"], downcast='float')
    df_train, df_valid = model_selection.train_test_split(dfs,
            test_size=0.1,
            random_state=42,
            stratify=dfs.label.values,
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    dataset_reader = dataset.Dataset()

    train_dataset = dataset_reader.read(df_train, return_pt=True)
    valid_sentence1, valid_sentence2, valid_labels = dataset_reader.read(df_valid)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch)
    # evaluator = evaluation.EmbeddingSimilarityEvaluator(valid_sentence1, valid_sentence2, valid_labels)
    evaluator = evaluation.BinaryClassificationEvaluator(valid_sentence1, valid_sentence2, valid_labels, batch_size=vaild_batch, show_progress_bar=False)

    word_embedding_model = models.Transformer(model_path, max_seq_length=max_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=max_length, activation_function=nn.Tanh())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

    train_loss = losses.CosineSimilarityLoss(model)

    engine.train(train_dataloader, model, train_loss, evaluator)


if __name__ == "__main__":
    run()
