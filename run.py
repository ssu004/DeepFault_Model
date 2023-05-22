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
from dfb.paramsampler import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import CSVLogger

import experiment
import shutil

import argparse

import scipy.io

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_seed = 5555

    parser = argparse.ArgumentParser("Experiment environment")
    parser.add_argument(
        "--data", required=True, dest="data"
    )
    parser.add_argument(
        "--model", required=True, dest="model"
    )
    parser.add_argument(
        "--optimizer", required=True, dest="optimizer"
    )
    parser.add_argument(
        "--epochs", required=True, dest="epochs"
    )
    parser.add_argument(
        "--batch_size", required=True, dest="batch_size"
    )
    parser.add_argument(
        "--gpus", required=True, dest="gpus"
    )
    parser.add_argument(
        "--exp_start", required=True, dest="exp_start"
    )
    parser.add_argument(
        "--exp_end", required=True, dest="exp_end"
    )
    args = parser.parse_args()

    data_name = args.data
    model_name = args.model
    optimizer_name = args.optimizer
    max_epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    num_workers = 10
    n_gpus = [int(args.gpus)]
    exps = [int(args.exp_start), int(args.exp_end)]

    exp_idx = np.arange(exps[0] ,exps[1] + 1)

    sample_length = experiment.get_sample_length(model_name)
    n_classes = experiment.get_num_classes(data_name)

    tf_data = experiment.get_transform(model_name)
    tf_label = [NpToTensor()]

    # sampler = experiment.get_sampler(optimizer_name, 512)
    hparams = scipy.io.loadmat(f"./hparam_list/{optimizer_name}.mat")
    del hparams["__header__"]
    del hparams["__version__"]
    del hparams["__globals__"]
    for k in hparams.keys():
        hparams[k] = hparams[k].ravel()

    train_loader, val_loader, test_loader = experiment.get_datamodule(data_name,
                                                                    sample_length,
                                                                    transforms.Compose(tf_data),
                                                                    transforms.Compose(tf_label),
                                                                    batch_size,
                                                                    n_classes)

    n_steps_d = len(train_loader.dataset) // (batch_size * len(n_gpus))

    snrs = [-4, -2, 0, 2, 4]

    exp_name = f"{data_name}_{model_name}_{optimizer_name}_{batch_size}[{exps[0]}-{exps[1]}]"


    if not os.path.isdir(f"./logs/{exp_name}"):
        os.mkdir(f"./logs/{exp_name}")
    else:
        shutil.rmtree(f"./logs/{exp_name}")
        os.mkdir(f"./logs/{exp_name}")

    result_log = open(f"./logs/{exp_name}/result.csv", "w")

    result_header = f"i,"
    for param in hparams.keys():
        result_header = result_header + f"{param},"
    result_header = result_header + "val_loss,val_acc,test_loss,test_acc,"
    for snr in snrs:
        result_header = result_header + f"{snr},"
    result_header = result_header + f"best_epoch\n"

    result_log.write(result_header)
    result_log.close()

    for i in exp_idx:
        result_log = open(f"./logs/{exp_name}/result.csv", "a")
        result_body = f"{i},"

        trials_name = f"{exp_name}/trials/{i}"
        # torch.manual_seed(model_seed)
        model_kwargs = {}
        if model_name == "ticnn":
            model_kwargs["device"] = f"cuda:{n_gpus[0]}"
        model = experiment.get_model(data_name, model_name, **model_kwargs)
        optimizer_kwargs = {}
        for hparam_name in hparams.keys():
            optimizer_kwargs[hparam_name] = hparams[hparam_name][i]
            result_body = result_body + f"{hparams[hparam_name][i]},"
        if optimizer_name == "adam":
            optimizer_kwargs["betas"] = (hparams["beta1"][i], hparams["beta2"][i])
            del optimizer_kwargs["beta1"]
            del optimizer_kwargs["beta2"]

        print(optimizer_kwargs)
        
        optimizer = experiment.get_optimizer(optimizer_name)(model.parameters(), **optimizer_kwargs)
        loss_fn = torch.nn.CrossEntropyLoss()

        logger = CSVLogger(f"./logs/{trials_name}/training", name="log")
        training_module = PlModule(model, optimizer, loss_fn, True)
        callback = ModelCheckpoint(monitor="val_loss",
                                dirpath=f"./logs/{trials_name}/best_model",
                                filename=f"model",
                                save_top_k=1,
                                mode="min")
        trainer = pl.Trainer(gpus=n_gpus,
                            max_epochs=max_epochs,
                            val_check_interval=n_steps_d,
                            default_root_dir=f"./logs/{trials_name}",
                            callbacks=[callback],
                            logger=logger)

        trainer.fit(model=training_module,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)
        training_module.load_from_checkpoint(f"./logs/{trials_name}/best_model/model.ckpt",
                                            model=model, optimizer=optimizer,
                                            loss_fn=loss_fn)
        result = trainer.test(model=training_module, dataloaders=test_loader)

        df_result = pd.read_csv(f"./logs/{trials_name}/training/log/version_0/metrics.csv")
        df_agg = df_result.groupby(by="epoch").agg("sum")
        df_agg.to_csv(f"./logs/{trials_name}/training/log/version_0/metrics.csv")
        df_best = df_agg[df_agg["val_loss"] == df_agg["val_loss"].min()]
        best_val_loss = df_best["val_loss"].values[0]
        best_val_acc = df_best["val_acc"].values[0]
        best_test_loss = result[0]['test_loss']
        best_test_acc = result[0]['test_acc']
        best_step = df_best.index.values[0]+1

        result_body = result_body + f"{best_val_loss},{best_val_acc},{best_test_loss},{best_test_acc},"

        snrs = [-4, -2, 0, 2, 4]
        _, _, noisy_array = experiment.get_data_array(data_name, sample_length)

        for snr in snrs:
            tf_noise = [AWGN(snr)] + tf_data
            noisy_loader = experiment.get_dataloader(noisy_array['data'],
                                                    noisy_array['label'],
                                                    transforms.Compose(tf_noise),
                                                    transforms.Compose(tf_label),
                                                    False,
                                                    batch_size,
                                                    num_workers)

            logger = CSVLogger(f"./logs/{trials_name}/snr_{snr}", name="log")
            trainer = pl.Trainer(gpus=n_gpus,
                                max_epochs=max_epochs,
                                val_check_interval=n_steps_d,
                                default_root_dir=f"./logs/{trials_name}/snr_{snr}",
                                callbacks=[callback], logger=logger)

            for _ in range(10):
                result = trainer.test(model=training_module, dataloaders=noisy_loader)
            
            noise_result = pd.read_csv(f"./logs/{trials_name}/snr_{snr}/log/version_0/metrics.csv")
            mean_acc = np.mean(noise_result["test_acc"])
            result_body = result_body + f"{mean_acc},"

        result_body = result_body + f"{best_step}\n"
        result_log.write(result_body)
        result_log.close()

