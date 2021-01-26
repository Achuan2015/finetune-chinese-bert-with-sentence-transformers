import pandas as pd
from sklearn import model_selection
from sentence_transformers import evaluation
from sentence_transformers import SentenceTransformer

import config
import dataset


def run():
    test_file = config.TEST_FILE
    test_batch = config.TEST_BATCH_SIZE
    model_save_path = config.MODEL_SAVE_PATH

    dfs = pd.read_csv(test_file, sep='\t', names=['idx', 'sent1', 'sent2', 'label'])
    dfs['label'] = pd.to_numeric(dfs['label'], downcast='float')

    dataset_reader = dataset.Dataset()
    test_sent1, test_sent2, test_labels = dataset_reader.read(dfs)
    
    evaluator = evaluation.BinaryClassificationEvaluator(test_sent1, test_sent2, test_labels, batch_size=test_batch, show_progress_bar=True)
    
    model = SentenceTransformer(model_save_path)
    model.evaluate(evaluator)


if __name__ == "__main__":
    run()
