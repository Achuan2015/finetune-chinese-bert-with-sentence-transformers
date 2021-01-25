import transformers

DEVICE = "cuda"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 32
EPOCH = 3
WARMUP_STEPS = 100
EVALUATION_STEPS = 500 
BERT_PATH = "../inputs/bert-base-chinese"
MODEL_SAVE_PATH = "../outputs/"
TRAINING_FILE = "../data/wz_train.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
        BERT_PATH,
        do_lower_case = True
)
