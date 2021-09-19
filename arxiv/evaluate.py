from spektral.data import SingleLoader


def evaluate_model(model, dataset, masks):
    loader_te = SingleLoader(dataset, sample_weights=masks["test"])
    eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
    return eval_results