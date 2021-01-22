import config

def train(dataloader, model, train_loss, evaluator=None):
    epochs = config.EPOCH
    warmup_steps = config.WARMUP_STEPS
    evaluation_steps = config.EVALUATION_STEPS
    model_save_path = config.MODEL_SAVE_PATH

    model.fit(
        train_objectives=[(dataloader, train_loss)], 
        epochs=epochs, 
        warmup_steps=warmup_steps, 
        evaluator=evaluator, 
        evaluation_steps=evaluation_steps,
        output_path=model_save_path)

def test(data_loader, model):
    pass
