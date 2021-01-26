import transformers

DEVICE = "cuda"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 64
TEST_BATCH_SIZE = 32
EPOCH = 5
WARMUP_STEPS = 10000
EVALUATION_STEPS = 5000 
BERT_PATH = "../inputs/bert-base-chinese"
MODEL_SAVE_PATH = "../outputs/hrtps/"
TRAINING_FILE = "../data/hrtps_taobao.csv"
TEST_FILE = "../data/hrtps_test_10000.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
        BERT_PATH,
        do_lower_case = True
)
