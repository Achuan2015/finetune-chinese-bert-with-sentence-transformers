from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
#cmodel = AutoModel.from_pretrained("bert-base-chinese")

tokenizer.save_pretrained('../inputs/bert-base-chinese')
#model.save_pretrained('models/bert-base-chinese')
