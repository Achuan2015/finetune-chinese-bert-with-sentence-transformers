from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sentence_transformers import evaluation
from torch.utils.data import DataLoader
from torch import nn


def run():
    word_embedding_model = models.Transformer('inputs/bert-base-chinese/', max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

    train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
               InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]

    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    # Evaluators
    sentences1 = ['This list contains the first column', 'With your sentences', 'You want your model to evaluate on']
    sentences2 = ['Sentences contains the other column', 'The evaluator matches sentences1[i] with sentences2[i]', 'Compute the cosine similarity and compares it to scores[i]']
    scores = [0.3, 0.6, 0.2]

    evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

    # Tune the model
    model_save_path = "outputs/"
    model.fit(
            train_objectives=[(train_dataloader, train_loss)], 
            epochs=1, 
            warmup_steps=100, 
            evaluator=evaluator, 
            evaluation_steps=500,
            output_path=model_save_path)
# model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)


if __name__ == "__main__":
    run()
