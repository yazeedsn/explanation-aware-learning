"""
Entry point for a single disease-classification run.

Example:
    python run_experiment.py

To run a study (e.g. sweep explanation-loss alpha, or compare score
formulations), build a list of `ExperimentConfig`s and loop over this
module's `run()` function -- nothing else needs to change.
"""

from torch import optim
from .vinbig_prep import open_dataset

from .core import (
    CombinedLoss, DiseaseTaskBuilder, ExperimentConfig,
    Trainer, build_dataloaders, build_model, plot_history,
)
from .core.seeding import set_seed


def run(config: ExperimentConfig):
    set_seed(config.train.seed)

    # Read the preprocessed data (disease list/order, image size, dtypes)
    dataset = open_dataset(config.data.processed_dir)

    # build disease specific splits train_ids, val_ids, test_ids for the specified disease conting positive and negative ids
    task_builder = DiseaseTaskBuilder(dataset, config.split)
    task = task_builder.build(config.disease)
    print(
        f"{config.disease}: train={len(task['train_ids'])}, "
        f"val={len(task['val_ids'])}, test={len(task['test_ids'])}"
    )

    disease_idx = dataset.metadata.disease_to_idx[config.disease]
    train_dl, val_dl, test_dl = build_dataloaders(task, dataset, disease_idx, config.train)

    device = config.train.resolve_device()
    model = build_model(config.model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    loss_fn = CombinedLoss(config.explanation_loss)

    trainer = Trainer(model, loss_fn, optimizer, None, config.train, config.model, config.run, config.disease)
    history = trainer.fit(train_dl, val_dl)

    plot_history(history, trainer.output_dir, show=False)
    return trainer, history


if __name__ == "__main__":
    config = ExperimentConfig(disease="Aortic enlargement")
    run(config)
