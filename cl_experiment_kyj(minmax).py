# 07/25 MinmaxScaling 추가
import os
import gdown
import hashlib
import zipfile

import os
from datetime import datetime
from dfb.download import *
from dfb.databuilder import *
from dfb.dataset import *
from dfb.processing import *
import experiment
import torchvision.transforms as transforms

import avalanche
import json
import argparse

from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.benchmarks.utils import make_classification_dataset
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from avalanche.training.supervised import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, JointTraining, EWC
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin, ReplayPlugin
from dfb.model.wdcnn import *

from dfb.model.wdcnn2 import *

model_name = "wdcnn2"

parser = argparse.ArgumentParser()

parser.add_argument(
    '--level',
    type=int,
    choices=[1, 2, 3],
    default=3
)
parser.add_argument(
    '--repeat',
    type=int,
    default=50
)
parser.add_argument(
    '--strategy',
    type=str,
    default='Replay'
)
parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
)
parser.add_argument(
    '--momentum',
    type=float,
    default=0
)
parser.add_argument(
    '--l2',
    type=float,
    dest='weight_decay',
    default=0.01,
)
parser.add_argument(
    '--device',
    choices=['cuda', 'cpu'],
    default='cuda',
)
parser.add_argument(
    '--batch-size',
    type=int,
    default=64,
)
parser.add_argument(
    '--epoch',
    type=int,
    default=100,
)

parser.add_argument(
    '--memory-size',
    type=int,
    default=200,
)

parser.add_argument(
    '--save_folder_name',
    type=str,
)

def generate_save_folder_name(model_name, strategy):
    # 현재의 날짜와 시간
    now = datetime.now()
    # 월과 일을 포맷으로 하는 문자열을 생성
    date_str = now.strftime("%m%d")
    # model_name을 대문자로 바꾸고, date_str 앞에 언더바를 추가
    save_folder_name = model_name.upper() + "_" + strategy.upper() + "_" + date_str

    return save_folder_name


def get_hash(filename):
    with open(filename, "rb") as f:
        data = f.read()
        hash = hashlib.md5(data).hexdigest()
    
    return hash

data_info = {
    "cwru": {"link": "https://drive.google.com/uc?id=1JfnCzisg0wTSkWw_I5sNLcvQMD5mloJy",
             "hash": "a66d9ea53e5b9959829c1d1057abc377"},
    "mfpt": {"link": "https://drive.google.com/uc?id=1HDmX9-v8dV1-53nvM9lSDj-2-S2_Dss5",
             "hash": "fcf44622538307e33503cb7f24fd33d3"},
    "ottawa": {"link": "https://drive.google.com/uc?id=1WelJO5RMFwKoNdumhtW-__PC881fh4J_",
               "hash": "ca0142f52e950b5579985586a6acc96a"
    }
}


if not os.path.isdir("./data"):
    os.mkdir("./data")

for key in data_info:
    filename = f"./data/{key}.zip"
    if not os.path.isfile(filename):
        gdown.download(data_info[key]["link"], f"./data/{key}.zip")
    else:
        hash = get_hash(filename)
        if hash != data_info[key]["hash"]:
            os.remove(filename)
            gdown.download(data_info[key]["link"], f"./data/{key}.zip")

for key in data_info:
    filename = f"./data/{key}.zip"
    zipfile.ZipFile(filename).extractall("./data/")

dfs = {}

for key in data_info:
    dfs[key] = download_data(f"./data/{key}", key)


# level 1 데이터셋 제작

df_cwru = dfs["cwru"]
df_mfpt = dfs["mfpt"]
df_ottawa = dfs["ottawa"]

data_level1 = {}

df_load1 = df_cwru[(df_cwru["load"] == 1) & (df_cwru["label"] != 999)]
df_load2 = df_cwru[(df_cwru["load"] == 2) & (df_cwru["label"] != 999)]
df_load3 = df_cwru[(df_cwru["load"] == 3) & (df_cwru["label"] != 999)]

data_level1["A"] = build_from_dataframe(df_load1, sample_length=4096, shift=2048)
data_level1["B"] = build_from_dataframe(df_load2, sample_length=4096, shift=2048)
data_level1["C"] = build_from_dataframe(df_load3, sample_length=4096, shift=2048)

# level 2 데이터셋 제작

def set_label_level2(row):
    label_map = {
        "N": 0,
        "B": 1,
        "IR": 2,
        "OR@06": 3
    }
    row["label"] = label_map[row["fault_type"]]
    return row

data_level2 = {}

df_normal = df_cwru[(df_cwru["fault_type"] == "N")]
df_007 = df_cwru[(df_cwru["crack_size"] == "007") & (df_cwru["label"] != 999)]
df_014 = df_cwru[(df_cwru["crack_size"] == "014") & (df_cwru["label"] != 999)]
df_021 = df_cwru[(df_cwru["crack_size"] == "021") & (df_cwru["label"] != 999)]

df_007 = pd.concat(objs=(df_normal, df_007)).apply(set_label_level2, axis="columns")
df_014 = pd.concat(objs=(df_normal, df_014)).apply(set_label_level2, axis="columns")
df_021 = pd.concat(objs=(df_normal, df_021)).apply(set_label_level2, axis="columns")

data_level2["A"] = build_from_dataframe(df_007, sample_length=4096, shift=2048)
data_level2["B"] = build_from_dataframe(df_014, sample_length=4096, shift=2048)
data_level2["C"] = build_from_dataframe(df_021, sample_length=4096, shift=2048)

# level 3 데이터셋 제작

data_level3 = {}

def set_label_level3(row):
    label_map = {
        "N": 0,
        "IR": 1,
        "OR@06": 2,
        "OR": 2
    }
    row["label"] = label_map[row["fault_type"]]
    return row

sample_map = {
    "cwru": {
        "0": 105,
        "1": 35,
        "2": 35
    },
    "mfpt": {
        "0": 140,
        "1": 60,
        "2": 42
    },
    "ottawa": {
        "0": 35,
        "1": 35,
        "2": 35
    }
}

filter_cwru = df_cwru[df_cwru["label"] != 999]
filter_cwru = filter_cwru[(df_cwru["fault_type"] == "N") | (df_cwru["fault_type"] == "IR") | (df_cwru["fault_type"] == "OR@06")].reset_index(drop=True)
filter_mfpt = df_mfpt[(df_mfpt["fault_type"] == "N") | (df_mfpt["fault_type"] == "IR") | (df_mfpt["fault_type"] == "OR")].reset_index(drop=True)
filter_ottawa = df_ottawa[(df_ottawa["fault_type"] == "N") | (df_ottawa["fault_type"] == "IR") | (df_ottawa["fault_type"] == "OR")].reset_index(drop=True)

filter_cwru = filter_cwru.apply(set_label_level3, axis="columns")
filter_mfpt = filter_mfpt.apply(set_label_level3, axis="columns")
filter_ottawa = filter_ottawa.apply(set_label_level3, axis="columns")

data_level3["A"] = bootstrap_from_dataframe(filter_cwru, 4096, 100, False, sample_map["cwru"])
data_level3["B"] = bootstrap_from_dataframe(filter_mfpt, 4096, 100, False, sample_map["mfpt"])
data_level3["C"] = bootstrap_from_dataframe(filter_ottawa, 4096, 100, False, sample_map["ottawa"])



class MinMaxScaling:
    def __init__(self, min, max, scale, symmetric) -> None:
        self.min = min
        self.max = max
        self.scale = scale
        self.symmetric = symmetric

    def __call__(self, x):
        if self.symmetric:
            return (x - self.min) / (self.max - self.min) * self.scale - (self.scale * 0.5)
        else:
            return (x - self.min) / (self.max - self.min) * self.scale

sample_length = experiment.get_sample_length(model_name)
tf_data = experiment.get_transform(model_name)
tf_data = transforms.Compose(tf_data)
tf_label = NpToTensor()
batch_size = 128
num_worker = 4

data_handler = DatasetHandler()

# ============================== level 1

(X_train, y_train), (X_val, y_val), (X_test, y_test) = \
    train_val_test_split(data_level1["A"][0], 
                         data_level1["A"][1], 
                         0.2, 0.2, 0.6, 42, 
                         True, True)

data_handler.assign(
    X_train, y_train, X_val, y_val, X_test, y_test,
     sample_length, "1A", tf_data, tf_label, batch_size, num_worker
)

(X_train, y_train), (X_val, y_val), (X_test, y_test) = \
    train_val_test_split(data_level1["B"][0], 
                         data_level1["B"][1], 
                         0.2, 0.2, 0.6, 42, 
                         True, True)

data_handler.assign(
    X_train, y_train, X_val, y_val, X_test, y_test,
     sample_length, "1B", tf_data, tf_label, batch_size, num_worker
)

(X_train, y_train), (X_val, y_val), (X_test, y_test) = \
    train_val_test_split(data_level1["C"][0], 
                         data_level1["C"][1], 
                         0.2, 0.2, 0.6, 42, 
                         True, True)

data_handler.assign(
    X_train, y_train, X_val, y_val, X_test, y_test,
     sample_length, "1C", tf_data, tf_label, batch_size, num_worker
)

# ============================== level 2

(X_train, y_train), (X_val, y_val), (X_test, y_test) = \
    train_val_test_split(data_level2["A"][0], 
                         data_level2["A"][1], 
                         0.2, 0.2, 0.6, 42, 
                         True, True)

data_handler.assign(
    X_train, y_train, X_val, y_val, X_test, y_test,
     sample_length, "2A", tf_data, tf_label, batch_size, num_worker
)

(X_train, y_train), (X_val, y_val), (X_test, y_test) = \
    train_val_test_split(data_level2["B"][0], 
                         data_level2["B"][1], 
                         0.2, 0.2, 0.6, 42, 
                         True, True)

data_handler.assign(
    X_train, y_train, X_val, y_val, X_test, y_test,
     sample_length, "2B", tf_data, tf_label, batch_size, num_worker
)

(X_train, y_train), (X_val, y_val), (X_test, y_test) = \
    train_val_test_split(data_level2["C"][0], 
                         data_level2["C"][1], 
                         0.2, 0.2, 0.6, 42, 
                         True, True)

data_handler.assign(
    X_train, y_train, X_val, y_val, X_test, y_test,
     sample_length, "2C", tf_data, tf_label, batch_size, num_worker
)

# ============================== level 3

(X_train, y_train), (X_val, y_val), (X_test, y_test) = \
    train_val_test_split(data_level3["A"][0], 
                         data_level3["A"][1], 
                         0.2, 0.2, 0.6, 42, 
                         True, True)

data_handler.assign(
    X_train, y_train, X_val, y_val, X_test, y_test,
     sample_length, "3A", tf_data, tf_label, batch_size, num_worker
)

(X_train, y_train), (X_val, y_val), (X_test, y_test) = \
    train_val_test_split(data_level3["B"][0], 
                         data_level3["B"][1], 
                         0.2, 0.2, 0.6, 42, 
                         True, True)

data_handler.assign(
    X_train, y_train, X_val, y_val, X_test, y_test,
     sample_length, "3B", tf_data, tf_label, batch_size, num_worker
)

(X_train, y_train), (X_val, y_val), (X_test, y_test) = \
    train_val_test_split(data_level3["C"][0], 
                         data_level3["C"][1], 
                         0.2, 0.2, 0.6, 42, 
                         True, True)

data_handler.assign(
    X_train, y_train, X_val, y_val, X_test, y_test,
     sample_length, "3C", tf_data, tf_label, batch_size, num_worker
)

def calcMinMax(dataset: Dataset):
    batch_min, batch_max = None, None
    
    for data, _ in dataset:
        sample_min, sample_max = torch.min(data).item(), torch.max(data).item()
        
        if not batch_min:
            batch_min = sample_min
        if not batch_max:
            batch_max = sample_max
        
        if sample_min < batch_min:
            batch_min = sample_min
        if sample_max > batch_max:
            batch_max = sample_max

    return batch_min, batch_max


def classes_in_level(level):
    if level == 1:
        return 10
    elif level == 2:
        return 4
    elif level == 3:
        return 3

class CLExperiment:
    
    def __init__(self, opt) -> None:

        self.opt = opt
        self.data_handler = data_handler
        self.avg_results = []

    def get_attr(self, name: str):
        if hasattr(self.opt, name):
            return getattr(self.opt, name)
        else:
            return None

    def _make_benchmark_with_level(self, data_handler: DatasetHandler, level: int):

        assert level in [1, 2, 3], "Benchmark level must be one of (1, 2, 3)"

        train_set_list = [data_handler.dataloaders[f'{level}{task}']['train'].dataset for task in ('A', 'B', 'C')]
        test_set_list = [data_handler.dataloaders[f'{level}{task}']['test'].dataset for task in ('A', 'B', 'C')]

        # 'make_classification_dataset' requires that the dataset has an attribute named 'targets'
        for train_set, test_set in zip(train_set_list, test_set_list):
            setattr(train_set, 'targets', train_set.label)
            setattr(test_set, 'targets', test_set.label)

        # AvalancheDatasets with task labels
        train_set_list = [make_classification_dataset(dataset, task_labels=idx) for idx, dataset in enumerate(train_set_list)]
        test_set_list = [make_classification_dataset(dataset, task_labels=idx) for idx, dataset in enumerate(test_set_list)]

        # Compose benchmark
        self.scenario = dataset_benchmark(
            train_set_list,
            test_set_list
        )

    def initialize(self, model, scenario=None, optimizer=None, criterion=None, eval_plugin=None):
        
        self.model = model
        level = self.get_attr('level')

        if not scenario:
            self._make_benchmark_with_level(self.data_handler, level)
        else:
            self.scenario = scenario

        self._initialize_trainig_stuff(self.model, optimizer, criterion)
        self._initialize_cl_strategy(eval_plugin)


    def _initialize_trainig_stuff(self, model: nn.Module, optimizer=None, criterion=None):

        lr = self.get_attr('lr')
        momentum = self.get_attr('momentum')
        weight_decay = self.get_attr('weight_decay')

        if not optimizer:
            optimizer = SGD(
                model.parameters(), lr=lr,
                momentum=momentum, weight_decay=weight_decay
            )
        self.optimizer = optimizer

        if not criterion:
            criterion = CrossEntropyLoss()
        self.criterion = criterion

    def _initialize_cl_strategy(self, eval_plugin):

        if not eval_plugin:
            eval_plugin = EvaluationPlugin(
                accuracy_metrics(epoch=True, experience=True, stream=True),
                loss_metrics(epoch=True, experience=True, stream=True),
                forgetting_metrics(experience=True, stream=True),
                loggers=self._initialize_loggers(self.get_attr('interactive'))
            )

        cl_type = self.get_attr('strategy').lower()

        common_args = dict(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            train_epochs=self.get_attr('epoch'),
            train_mb_size=self.get_attr('batch_size'),
            eval_mb_size=self.get_attr('batch_size'),
            device=self.get_attr('device'),
            evaluator=eval_plugin
        )

        if cl_type == 'naive':
            self.strategy = Naive(**common_args)
        elif cl_type == 'replay':
            self.strategy = Replay(
                **common_args,
                mem_size=self.get_attr('memory_size')
            )
        elif cl_type == 'joint':
            self.strategy = JointTraining(**common_args)
        elif cl_type == 'cumulative':
            self.strategy = Cumulative(**common_args)

        elif cl_type == 'cwrstar':
            self.strategy = CWRStar(
                cwr_layer_name='head',   ####### WDCNN2 아키텍처 기준
                **common_args
            )
        elif cl_type == 'gdumb':
            self.strategy = GDumb(
                **common_args
            )
        elif cl_type == 'lwf':
            self.strategy = LwF(
                alpha=0.2,  
                temperature=1.2,  
                **common_args
            )
        elif cl_type == 'gem':
            self.strategy = GEM(
                patterns_per_exp=1024,  # 내가 임의로 설정함
                **common_args
            )
        elif cl_type == 'ewc':
            self.strategy = EWC(
                ewc_lambda=0.1,  # 1보다 작게 해야함
                **common_args
            )
        else:
            raise NotImplementedError(f"CL strategy '{cl_type}' has not been implemented yet!!!")


    def _make_param_string(self):
        model_name = type(self.model).__name__
        level = str(self.get_attr('level'))
        strategy = self.get_attr('strategy')
        
        optimizer = type(self.optimizer).__name__
        lr = self.get_attr('lr')
        weight_decay = self.get_attr('weight_decay')
        momentum = self.get_attr('momentum')

        return f'{model_name}_level_{level}_{strategy}_{optimizer}_lr_{lr}_momentum_{momentum}_l2_reg_{weight_decay}'

    def _make_log_path(self):
        save_folder = self.get_attr('save_folder_name')
        os.makedirs(save_folder, exist_ok=True)
        log_path = os.path.join(save_folder, self._make_param_string() + '.log')  
        return log_path

    def _initialize_loggers(self, interactive=True, text=True,):
        loggers = []
        if interactive:
            loggers.append(InteractiveLogger())
        if text:
            log_path = self._make_log_path()
            loggers.append(TextLogger(open(log_path, 'a')))
        self.loggers = loggers
        return self.loggers


    def _get_exp_metric_key(self, metric: str, phase: str, task: int, exp: int=None):

        if not exp:
            exp = task

        if metric.lower() == 'acc':
            header = 'Top1_Acc_Exp'
        elif metric.lower() == 'loss':
            header = 'Loss_Exp'
        elif metric.lower() == 'forgetting':
            header = 'ExperienceForgetting'

        return f'{header}/{phase}_phase/test_stream/Task{task:>03d}/Exp{exp:>03d}'


    def _get_metrics_for_joint(self, result):
        metrics = dict()
        avg_accuracy = [result[-1][self._get_exp_metric_key('acc', 'eval', task)] for task in range(3)]
        avg_forgetting = [result[-1][self._get_exp_metric_key('forgetting', 'eval', task)] for task in range(3)]

        metrics['avg_accuracy'] = avg_accuracy
        metrics['avg_forgetting'] = avg_forgetting

        return metrics


    def _get_avg_metrics(self, result):
        if self.get_attr('strategy') == 'joint':
            return self._get_metrics_for_joint()

        num_tasks = len(result)

        metrics = dict()
        avg_accuracy = []
        avg_forgetting = []
        for current in range(num_tasks):
            accuracies = [result[current][self._get_exp_metric_key('acc', 'eval', past)] for past in range(current+1)]
            forgettings = [result[current][self._get_exp_metric_key('forgetting', 'eval', past)] for past in range(current)]

            avg_accuracy.append(np.mean(accuracies))
            if forgettings:
                avg_forgetting.append(np.mean(forgettings))
            else:
                avg_forgetting.append(0)

        metrics['avg_accuracy'] = avg_accuracy
        metrics['avg_forgetting'] = avg_forgetting

        return metrics


    def _get_text_logger(self):

        for logger in self.loggers:
            if isinstance(logger, TextLogger):
                return logger
        return None
    

    def _append_results(self, results):
        avg_metrics = self._get_avg_metrics(results)
        self.avg_results.append(avg_metrics)
        
        print(avg_metrics)

        text_logger = self._get_text_logger()
        if text_logger:
            print(avg_metrics, file=text_logger.file, flush=True)


    def save_results(self, filename=None):
        if not filename:
            filename = self._make_param_string() + '_results.json'
        
        save_folder = self.get_attr('save_folder_name')
        os.makedirs(save_folder, exist_ok=True)
        
        filepath = os.path.join(save_folder, filename)
        json.dump(self.avg_results, open(filepath, 'w'), indent=4)

    def get_best_result(self):

        final_avg_acc = [metrics['avg_accuracy'][-1] for metrics in self.avg_results]
        best_result_idx = np.argmax(final_avg_acc)
        
        return best_result_idx, self.avg_results[best_result_idx]


    def _execute_joint_training(self):

        results = []

        self.strategy.train(self.scenario.train_stream)
        results.append(self.strategy.eval(self.scenario.test_stream))
        
        self._append_results(results)


    def _execute(self):

        results = []
        
        for exp in self.scenario.train_stream:
            self.strategy.train(exp)
            results.append(self.strategy.eval(self.scenario.test_stream))

        self._append_results(results)


    def execute(self, exec_id=0):
        for logger in self.loggers:
            if isinstance(logger, TextLogger):
                print(f"Start execution {exec_id}!!!", file=logger.file, flush=True)

        if self.get_attr('strategy') == 'joint':
            self._execute_joint_training()
        else:
            self._execute()


opt = parser.parse_args()
if opt.save_folder_name is None:
    opt.save_folder_name = generate_save_folder_name(model_name, opt.strategy)

cl_experiment = CLExperiment(opt)

repeat = cl_experiment.get_attr('repeat')
level = cl_experiment.get_attr('level')
for idx in range(repeat):
    model = WDCNN2(n_classes=classes_in_level(level))
    
    cl_experiment.initialize(model)
    cl_experiment.execute(idx)

cl_experiment.save_results()

########## 자동완성 ############
best_result_idx, best_result = cl_experiment.get_best_result()

print(f"Best result is {best_result_idx}th execution with {best_result['avg_accuracy'][-1]} accuracy!!!")





