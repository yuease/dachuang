from pathlib import Path
import argparse
import yaml
import torch
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn

import shutil


from train import BasicTrain

from model.model import  KMGCN
from dataloader import init_dataloader

def get_files_in_directory(directory):
    files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            files.append(filepath)
    return files

def main(args):
    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)
        print('config:', config)

    all_acc = []

    # ========= 10-fold CV =========
    for fold in range(10):
        print(f"\n================ Fold {fold} ================\n")
        config['data']['fold'] = fold

        # ---------- Data ----------
        dataloaders, node_size, node_feature_size, timeseries_size, datasets = init_dataloader(
            config['data'], fold=fold
        )

        config['train']["seq_len"] = timeseries_size
        config['train']["node_size"] = node_size

        # ---------- Model ----------
        class_dim = 2
        model = KMGCN(config, node_size, node_feature_size, timeseries_size, class_dim)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay']
        )
        opts = (optimizer,)

        save_folder_name = (
            Path(config['train']['log_folder']) /
            Path(config['model']['name']) /
            Path(f"{config['data']['dataset']}_{config['data']['atlas']}/fold_{fold}")
        )

        train_process = BasicTrain(
            config['train'],
            model,
            opts,
            dataloaders,
            save_folder_name,
            datasets
        )

        # ---------- Train ----------
        train_process.train()
        print(f"[Fold {fold}] Best val acc:", train_process.best_acc_val)

        # ---------- Test ----------
        accs = []
        directory_path = 'best_model/nc_ad'
        files = get_files_in_directory(directory_path)

        for path in files:
            acc = train_process.test(path)
            accs.append(acc)

        best_acc = max(accs)
        all_acc.append(best_acc)

        print(f"[Fold {fold}] Test acc: {best_acc:.4f}")

        # clean
        for file in files:
            os.remove(file)

    # ========= Final Result =========
    all_acc = np.array(all_acc)
    print("\n================ Final 10-fold Result ================")
    print("Acc per fold:", all_acc)
    print("Mean Acc: {:.4f}".format(all_acc.mean()))
    print("Std  Acc: {:.4f}".format(all_acc.std()))

        

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='setting/abide_PLSNet.yaml', type=str,
                        help='Configuration filename for training the model.')
    parser.add_argument('--repeat_time', default=100, type=int)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

    main(args)
