from ray import tune
import pytorch_lightning as pl 
from pytorch_lightning.loggers import TensorBoardLogger
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune import CLIReporter
from gesticulator.model.model import GesticulatorModel
from gesticulator.config.model_config import construct_model_config_parser
import numpy as np
from os import path
def add_training_script_arguments(parser):
    parser.add_argument('--save_model_every_n_epochs', '-ckpt_freq', type=int, default=0,
                        help="The frequency of model checkpoint saving.")
    parser.add_argument('--use_mirror_augment', '-mirror', action='store_true',
                        help="If set, use auxiliary mirrored motion dataset")
    return parser

def train_model(hparams, num_epochs, num_gpus):
    model = GesticulatorModel(hparams)
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=TensorBoardLogger(
            save_dir=model.hparams.result_dir, name="", version="."),
            #progress_bar_refresh_rate=1,
            callbacks=[TuneReportCallback(
                {
                    "loss": "train/full_loss"
                }, on="epoch_end"
            )]
        )
    trainer.fit(model)

def tune_hparam_search(hparams, num_samples, num_epochs, gpus_per_trial):
    # List of tuned hyperparams
    config = \
    {
        "batch_size": tune.choice([32, 64, 128, 256]),
        "vel_coef": tune.sample_from(
            lambda spec: 0.1 ** (np.random.randint(3, 10))),
        "learning_rate": tune.sample_from(
            lambda _: 10 ** (- np.random.randint(3, 6))),
        "dropout": tune.sample_from(
            lambda _: np.random.randint(30) / 100),
        "dropout_multiplier": tune.choice([1, 2, 3]),
        "speech_enc_frame_dim": tune.sample_from(
            lambda _: np.random.randint(2, 15) * 10),
        "full_speech_enc_dim": tune.sample_from(
            lambda _: np.random.randint(3, 9) * 124),
        "n_layers": tune.randint(1, 4),
        "first_l_sz": tune.choice([128, 256, 384, 512]),
        "second_l_sz": tune.choice([128, 256, 384, 512]),
        "third_l_sz": tune.choice([128, 256, 384, 512]),
        "n_prev_poses": tune.sample_from(
            lambda _: np.random.randint(1,4)),
    }
    hparams.update(config)

    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["loss", "training_iteration"])

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)


    analysis = tune.run(
        tune.with_parameters(
            train_model,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial
        ),        
        config=hparams,
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="gesticulator_hparam_search",
        local_dir=path.join(hparams["save_dir"], hparams["run_name"])
    )
    
    print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == '__main__':
    # Model parameters are added here
    parser = construct_model_config_parser()
    
    # Add training-script specific parameters
    parser = add_training_script_arguments(parser) 

    hyperparams = parser.parse_args()
    # convert to dictionary
    hyperparams = vars(hyperparams)
    root_dir = "/home/work/Desktop/repositories/gesticulator/"
    hyperparams["data_dir"] = path.join(root_dir, "dataset/processed")
    hyperparams["result_dir"] = path.join(root_dir, "results")
    hyperparams["utils_dir"] = path.join(root_dir, "gesticulator/utils")

    tune_hparam_search(hyperparams, 
        num_samples=2, num_epochs=1, gpus_per_trial=0)

