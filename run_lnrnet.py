# 연구재단 과제 모델만들기 실험
from dfb.download import *
from dfb.databuilder import *
from dfb.dataset import *
from dfb.trainmodule import *
from dfb.processing import *
from dfb.model.ticnn import *
from dfb.model.wdcnn import *
from dfb.model.stimcnn import *
from dfb.model.stftcnn import *
from dfb.model.wdcnnrnn import *
from dfb.model.aannet import *
from dfb.model.nrf_model import *
from dfb.paramsampler import *

import experiment
from sklearn.model_selection import train_test_split
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import CSVLogger

import argparse

def run_exp(
    exp_name, gpu_name, model_name
):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = 128
    n_workers = 10
    sample_length = experiment.get_sample_length(model_name)
    n_epochs = 200
    trials = 30
    gpu = gpu_name

    tf_data = experiment.get_transform(model_name)
    tf_label = [NpToTensor()]
    tf_noise = [AWGN(0.0)] + tf_data

    data_tags = ["D", "E"]

    domain_sign = {
        "D": ["D", "Dn"],
        "E": ["E", "En"],
    }

    dmodule = DatasetHandler()

    for tag in data_tags:
        train_data = np.load(f"./nrf_dataset/{tag}/train.npz")
        X_train = train_data["data"]
        y_train = train_data["label"]

        val_data = np.load(f"./nrf_dataset/{tag}/val.npz")
        X_val = val_data["data"]
        y_val = val_data["label"]

        test_data = np.load(f"./nrf_dataset/{tag}/test.npz")
        X_test = test_data["data"]
        y_test = test_data["label"]

        dmodule.assign(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            sample_length,
            tag,
            transforms.Compose(tf_data),
            transforms.Compose(tf_label),
            batch_size,
            n_workers,
        )

        if tag == "D":
            dmodule.assign(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                sample_length,
                "Dn",
                transforms.Compose(tf_noise),
                transforms.Compose(tf_label),
                batch_size,
                n_workers,
            )
        if tag == "E":
            dmodule.assign(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                sample_length,
                "En",
                transforms.Compose(tf_noise),
                transforms.Compose(tf_label),
                batch_size,
                n_workers,
            )

    for data_tag in data_tags:
        for i in range(trials):
            model_kwarg = {}
            if model_name == "ticnn":
                model_kwarg['device'] = f"cuda:{gpu}"
            if data_tag == "E":
                model = experiment.get_model(data="mfpt", model=model_name, **model_kwarg)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            else:
                model = experiment.get_model(data="cwru", model=model_name, **model_kwarg)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            loss_fn = torch.nn.CrossEntropyLoss()

            n_steps_d = len(dmodule.dataloaders[data_tag]["train"].dataset) // (128)

            logger = CSVLogger(f"{exp_name}/{data_tag}/trials/{i}", name="log")
            module = PlModule(model, optimizer, loss_fn, True)
            callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=f"{exp_name}/{data_tag}/trials/{i}/best_model",
                filename=f"model",
                save_top_k=1,
                mode="min",
            )
            trainer = pl.Trainer(
                gpus=[gpu],
                max_epochs=n_epochs,
                val_check_interval=n_steps_d,
                default_root_dir=f"{exp_name}",
                callbacks=[callback],
                logger=logger,
            )

            trainer.fit(
                model=module,
                train_dataloaders=dmodule.dataloaders[data_tag]["train"],
                val_dataloaders=dmodule.dataloaders[data_tag]["val"],
            )

    exp_result = {
        "trials": [],
        "source": [],
        "target": [],
        "test_acc": [],
        "test_loss": [],
    }

    for data_tag in data_tags:
        for i in range(trials):
            model_kwarg = {}
            if model_name == "ticnn":
                model_kwarg['device'] = f"cuda:{gpu}"
            if data_tag == "E":
                model = experiment.get_model(data="mfpt", model=model_name, **model_kwarg)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            else:
                model = experiment.get_model(data="cwru", model=model_name, **model_kwarg)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            loss_fn = torch.nn.CrossEntropyLoss()

            n_steps_d = len(dmodule.dataloaders[data_tag]["train"].dataset) // (128)

            logger = CSVLogger(f"{exp_name}/{data_tag}/trials/{i}", name="log")
            module = PlModule(model, optimizer, loss_fn, True)
            module.load_from_checkpoint(
                f"{exp_name}/{data_tag}/trials/{i}/best_model/model.ckpt",
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
            )
            callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=f"{exp_name}/{data_tag}/trials/{i}/best_model",
                filename=f"model",
                save_top_k=1,
                mode="min",
            )
            trainer = pl.Trainer(
                gpus=[gpu],
                max_epochs=n_epochs,
                val_check_interval=n_steps_d,
                default_root_dir=f"{exp_name}",
                callbacks=[callback],
                logger=logger,
            )

            for domain in domain_sign[data_tag]:
                result = trainer.test(
                    model=module, dataloaders=dmodule.dataloaders[domain]["test"]
                )

                exp_result["trials"].append(i)
                exp_result["source"].append(data_tag)
                exp_result["target"].append(domain)
                exp_result["test_acc"].append(result[0]["test_acc"])
                exp_result["test_loss"].append(result[0]["test_loss"])

    exp_result_df = pd.DataFrame(exp_result)
    exp_result_df.to_csv(f"{exp_name}/result.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Exp env")
    parser.add_argument("--gpu", required=True, dest="n_gpu")
    parser.add_argument("--model", required=True, dest="model")
    args = parser.parse_args()
    n_gpu = int(args.n_gpu)
    model_name = args.model

    exp_name = f"lnrnet_logs/{model_name}"
    run_exp(exp_name, n_gpu, model_name)