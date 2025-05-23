import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from caffe2.python.examples.imagenet_trainer import Train
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random
import scipy
import collections
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
from sklearn.metrics import roc_auc_score,precision_recall_curve, auc
import time

from models.SocialInference import Regession_Baseline,Regression_Dataset
from models.LocalMatching import Spatial_Initiate,SOM
from models.GlobalMatching import Regression_Dataset_Constractive,Global_Tuning
from utils.load_datasets import load_dataset
from utils.utils import graph_construction,location_entropy_construction,sample_anchors,test_foundation_model

torch.manual_seed(0)
random.seed(0)

def Train_USRC(foundation_model_trained_on, device, feature_dim=256,
             batch_size=256,  additional_information=None, epochs = 50,    noise_scale = 0.00):
    if 'Max' in additional_information:
        pooling_method = 'Max'
    elif 'Attention' in additional_information:
        pooling_method = 'Attention'
    elif 'Mix' in additional_information:
        pooling_method = 'Mix'
    else:
        pooling_method = 'Mean'

    uid_time_checkin, colocation, loc2index = load_dataset(foundation_model_trained_on)

    random.seed(0)
    torch.manual_seed(0)
    # results=random.sample(results, round(len(results)*0.1))
    random.shuffle(colocation)
    num_locs = len(loc2index)
    print('num_loc ', num_locs)
    training = colocation[:round(len(colocation) * 0.6)]
    testing = colocation[round(len(colocation) * 0.6):]
    dataset_training = Regression_Dataset(training, loc2index=loc2index, user_checkin=uid_time_checkin, augmentation=0)
    dataset_testing = Regression_Dataset(testing, loc2index=loc2index, user_checkin=uid_time_checkin, augmentation=0)
    print(len(dataset_training), len(dataset_testing))

    train_loader = DataLoader(dataset_training, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=False, collate_fn=dataset_training.collate_fn_pad)
    test_loader = DataLoader(dataset_testing, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                             drop_last=False, collate_fn=dataset_training.collate_fn_pad)


    pooling = pooling_method
    save_path = '../data/Regession_Baseline_Noise00_MixPooling_'+str(foundation_model_trained_on)+'_MSEnoAug_x100_TrajSameWeight.pth'
    model = Regession_Baseline(num_loc=num_locs, loc_embedding=feature_dim, week_embedding=6, hour_embedding=6,
                               feature_dim=feature_dim).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = torch.nn.MSELoss()  # BCEWithLogitsLoss() # 
    best_result = 0

    for e in range(epochs):
        training_loss = []
        model.train()
        train_bar = tqdm(train_loader)
        for x, times, covisited, covisited_time, traj1, traj2, y in train_bar:
            optimizer.zero_grad()
            x, times, covisited, covisited_time, traj1, traj2, y = x.to(device), times.to(device), covisited.to(
                device), covisited_time.to(device), traj1.to(device), traj2.to(device), y.to(device) * 100
            prediction = model(x, times, covisited, covisited_time, traj1, traj2, training=True, pooling=pooling,
                               noise_scale=noise_scale)
            loss = loss_fn(prediction, y)
            loss.backward()
            optimizer.step()
            training_loss.append(torch.sum(loss).cpu().detach())

        if e % 5 == 0:
            with torch.no_grad():
                model.eval()
                labels = []
                predictions = []
                xs = []
                test_bar = tqdm(test_loader)
                for x, times, covisited, covisited_time, traj1, traj2, y in test_bar:
                    x, times, covisited, covisited_time, traj1, traj2, y = x.to(device), times.to(device), covisited.to(
                        device), covisited_time.to(device), traj1.to(device), traj2.to(device), y.to(device)
                    prediction = model(x, times, covisited, covisited_time, traj1, traj2, training=False,
                                       pooling=pooling, noise_scale=noise_scale)
                    predictions.append(prediction.detach().cpu().view(-1).tolist())
                    labels.append(y.detach().cpu().view(-1).tolist())
                    x_cleaned = [seq[seq != 0].tolist() for seq in x]  # Remove padding and convert to list
                    xs.append(x_cleaned)

                labels = sum(labels, [])
                predictions = sum(predictions, [])
                roc_auc = roc_auc_score(labels, predictions)
                precision, recall, _ = precision_recall_curve(labels, predictions)
                pr_auc = auc(recall, precision)
                print(f"PR AUC: {pr_auc:.4f}")
                print('Area under the ROC curve:', roc_auc)

                if pr_auc > best_result and e > 0:
                    best_result = pr_auc
                    best_model = model.state_dict()
                    torch.save(best_model, save_path)

        print(e, ' epoch ', np.mean(training_loss))  # , np.mean(training_loss)


if __name__ == '__main__':

    foundation_model_trained_on_list = ['Gowalla_LA', 'Gowalla_NY','Gowalla_ST', 'Foursquare_LA', 'Foursquare_NY', ]  # ,'Foursquare_LA'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = 256
    batch_size = 256  # 256  #len(dataset_training)-1#(len(dataset_training)//2)-1
    additional_information='Mix'
    
    for foundation_model_trained_on in foundation_model_trained_on_list:
        Train_USRC(foundation_model_trained_on, device,additional_information=additional_information, feature_dim=feature_dim,batch_size=batch_size)





