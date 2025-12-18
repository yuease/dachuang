import numpy as np
import torch
import torch.utils.data as utils
import os


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
def init_dataloader(dataset_config, fold=0):
    data = np.load(dataset_config["time_seires"], allow_pickle=True).item()                            

    final_fc = data["timeseires"]                                                              
    final_pearson = data["corr"]                                                               
    final_pcorr = data["pcorr"]
    labels = data["label"]                                                                   
    ages = data["age"]
    genders = data["gender"]

    _, _, timeseries = final_fc.shape                           

    _, node_size, node_feature_size = final_pearson.shape      

    scaler = StandardScaler(mean=np.mean(final_fc), std=np.std(final_fc))
    
    final_fc = scaler.transform(final_fc)                   


    pseudo = []     
    for i in range(len(final_fc)):
        pseudo.append(np.diag(np.ones(final_pearson.shape[1])))     


    if 'cc200' in  dataset_config['atlas']:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 200, 200))             
    elif 'aal' in dataset_config['atlas']:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 116, 116))
    elif 'cc400' in dataset_config['atlas']:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 392, 392))
    else:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 111, 111))

    final_fc, final_pearson, final_pcorr, labels, pseudo_arr, ages, genders = [torch.from_numpy(d).float() for d in (final_fc, final_pearson, final_pcorr, labels, pseudo_arr, ages, genders)]      # 转为torch向量
    # ====== DEBUG: 检查 ABIDE 数据维度 ======
    #print("DEBUG shapes:")
    #print("final_fc:", final_fc.shape)
    #print("final_pearson:", final_pearson.shape)
    #print("final_pcorr:", final_pcorr.shape)
    #print("labels:", labels.shape)
    #print("ages:", ages.shape)
    #print("genders:", genders.shape)
    #print("pseudo:", pseudo_arr.shape)
    dataset = utils.TensorDataset(
    final_fc,
    final_pearson,
    final_pcorr,
    labels,
    pseudo_arr,
    ages,
    genders
    )

    # ===============================
    # 使用固定的 10-fold indices
    # ===============================
    fold_dir = dataset_config.get("fold_dir", "fold_indices")

    train_idx = np.load(os.path.join(fold_dir, f"fold_{fold}_train.npy"))
    val_idx   = np.load(os.path.join(fold_dir, f"fold_{fold}_test.npy"))

    train_dataset = utils.Subset(dataset, train_idx)
    val_dataset   = utils.Subset(dataset, val_idx)


    train_dataloader = utils.DataLoader(
    train_dataset,
    batch_size=dataset_config["batch_size"],
    shuffle=True,
    drop_last=True
    )

    val_dataloader = utils.DataLoader(
    val_dataset,
    batch_size=dataset_config["batch_size"],
    shuffle=False,
    drop_last=False
    )



    return (train_dataloader, val_dataloader), node_size, node_feature_size, timeseries, (train_dataset, val_dataset)
