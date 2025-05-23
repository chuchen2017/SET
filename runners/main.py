import random

from models.SocialInference import Regession_Baseline,Regression_Dataset
from models.LocalMatching import Spatial_Initiate,SOM
from models.GlobalMatching import Regression_Dataset_Constractive,Global_Tuning
from utils.load_datasets import load_dataset
from utils.utils import graph_construction,location_entropy_construction,sample_anchors,test_foundation_model
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
import gc
from torch.utils.data import random_split

torch.manual_seed(0)
random.seed(0)

def load_logfile():
    if os.path.exists('../data/result_log.json'):
        with open('../data/result_log.json', 'r', encoding='utf-8') as file:
            result_log = json.loads(file.read())
    else:
        result_log = {}
        with open('../data/result_log.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(result_log))
    return result_log

def transfer(foundation_model_trained_on, city_apply_to, device, foundation_model_path, feature_dim=256,
             test_batch_size=256, contrastive_batch_size=4, load_SpatialInitiate=False, additional_information=None,
             contrastive_tuning_epochs=3,contrastive_tuning_temp=100, anchor_rate=0.8,top_n=500):
    if 'Max' in additional_information:
        pooling_method = 'Max'
    elif 'Attention' in additional_information:
        pooling_method = 'Attention'
    elif 'Mix' in additional_information:
        pooling_method = 'Mix'
    else:
        pooling_method = 'Mean'

    uid_time_checkin_trainedOn, colocation_trainedOn, loc2index_trainedOn = load_dataset(foundation_model_trained_on)
    uid_time_checkin_test, colocation_test, loc2index_test = load_dataset(city_apply_to)

    loc2index_LA = loc2index_trainedOn
    loc2index_NY = loc2index_test
    uid_time_checkinLA = uid_time_checkin_trainedOn
    uid_time_checkinNY = uid_time_checkin_test
    resultsLA = colocation_trainedOn
    resultsNY = colocation_test
    dataset_testing = Regression_Dataset(resultsNY, user_checkin=uid_time_checkinNY, loc2index=loc2index_NY,
                                         augmentation=0)
    test_loader_NY = DataLoader(dataset_testing, batch_size=test_batch_size, shuffle=False, num_workers=16,
                                pin_memory=True,
                                drop_last=False, collate_fn=dataset_testing.collate_fn_pad)

    print('Number of location in Foundation model ', len(loc2index_trainedOn))
    print('Number of location in City apply to ', len(loc2index_test))
    # location_entropy_LA = location_entropy_construction(loc2index_LA, uid_time_checkinLA,colocation=resultsLA)
    # print(len(location_entropy_LA))
    # gaussian_graph_LA = graph_construction(loc2index_LA, uid_time_checkinLA,colocation=resultsLA)
    try:
        gaussian_graph_NY = torch.load('../data/' + city_apply_to + '_graph.pth')
    except:
        gaussian_graph_NY = graph_construction(loc2index_NY, uid_time_checkinNY)
        torch.save(gaussian_graph_NY, '../data/' + city_apply_to + '_graph.pth')

    try:
        gaussian_graph_LA = torch.load('../data/' + foundation_model_trained_on + '_graph.pth')
    except:
        gaussian_graph_LA = graph_construction(loc2index_LA, uid_time_checkinLA) #,colocation=resultsLA
        torch.save(gaussian_graph_LA,
                   '../data/' + foundation_model_trained_on + '_graph.pth')

    print('Done loading Graphs')

    try:
        with open('../data/' + city_apply_to + '_location_entropy.json', 'r',
                  encoding='utf-8') as file:
            location_entropy_NY = json.load(file.read())
    except:
        location_entropy_NY = location_entropy_construction(loc2index_NY, uid_time_checkinNY)
        with open('../data/' + city_apply_to + '_location_entropy.json', 'w',
                  encoding='utf-8') as file:
            file.write(json.dumps(location_entropy_NY))

    try:
        with open('../data/' + foundation_model_trained_on + '_location_entropy.json', 'r',
                  encoding='utf-8') as file:
            location_entropy_LA = json.load(file.read())
    except:
        location_entropy_LA = location_entropy_construction(loc2index_LA, uid_time_checkinLA,colocation=resultsLA)
        with open('../data/' + foundation_model_trained_on + '_location_entropy.json', 'w',
                  encoding='utf-8') as file:
            file.write(json.dumps(location_entropy_LA))

    print('Done loading Location Entropy')


    num_anchor_points = round(min(len(location_entropy_NY), len(location_entropy_LA))*anchor_rate)
    anchor_LA, anchor_NY = sample_anchors(location_entropy_LA, location_entropy_NY, num_anchor_points=num_anchor_points, top_n=top_n)
    anchor_NY2LA = dict(zip(anchor_NY, anchor_LA))
    anchor_LA2NY = dict(zip(anchor_LA, anchor_NY))

    prediction_model = Regession_Baseline(num_loc=len(loc2index_LA), loc_embedding=feature_dim, week_embedding=6,
                                          hour_embedding=6, feature_dim=feature_dim).double().to(device)
    state_dict = torch.load(foundation_model_path, map_location=device)
    prediction_model.load_state_dict(state_dict)
    prediction_model.eval()
    embedding_LA = torch.nn.Embedding(num_embeddings=len(loc2index_LA), embedding_dim=feature_dim,
                                      padding_idx=0).double().to(device)
    embedding_LA.weight.data = prediction_model.loc_embedding.weight.data.clone().detach()

    prediction_model.loc_embedding.weight.data = torch.nn.Embedding(num_embeddings=len(loc2index_NY),
                                                                    embedding_dim=feature_dim,
                                                                    padding_idx=0).double().to(device).weight.data
    prediction_model.eval()
    pr_auc, roc_auc = test_foundation_model(prediction_model, test_loader_NY, pooling_method)
    print('Randomly Initiate Embedding Result: ')
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"AU ROC: {roc_auc:.4f}")

    result_log = load_logfile()
    if additional_information is None:
        result_log[city_apply_to + '_' + foundation_model_trained_on + '_Random'] = (pr_auc, roc_auc)
    else:
        result_log[city_apply_to + '_' + foundation_model_trained_on + '_Random_' + additional_information] = (
            pr_auc, roc_auc)
    with open('../data/result_log.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(result_log))

    embedding_file_path = '../data/embedding' + city_apply_to + '_' + foundation_model_trained_on + '_SpatialInitiate.pth'
    if os.path.exists(embedding_file_path) and load_SpatialInitiate == True: #
        embedding_NY = torch.load(embedding_file_path, map_location=device)
        print("Loaded Spatial Embedding from file.")
    else:
        #embedding_LA,loc2index_LA,gaussian_graph_LA,anchor_LA,loc2index_NY,gaussian_graph_NY,anchor_NY,anchor_NY2LA
        embedding_NY = Spatial_Initiate(embedding_LA, loc2index_LA, gaussian_graph_LA, anchor_LA, loc2index_NY,
                                        gaussian_graph_NY, anchor_NY, anchor_NY2LA, device,
                                        noise_level = 0.5,top_k = 5)
        torch.save(embedding_NY, embedding_file_path)
        print("Saved Spatial Embedding to file.")

    prediction_model.loc_embedding.weight.data = embedding_NY.weight.data.double()
    pr_auc, roc_auc = test_foundation_model(prediction_model, test_loader_NY, pooling_method)
    print('Spatial Initiate Embedding Result: ')
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"AU ROC: {roc_auc:.4f}")
    result_log = load_logfile()
    if additional_information is None:
        result_log[city_apply_to + '_' + foundation_model_trained_on + '_SpatialInitiate'] = (pr_auc, roc_auc)
    else:
        result_log[city_apply_to + '_' + foundation_model_trained_on + '_SpatialInitiate_' + additional_information] = (
            pr_auc, roc_auc)
    with open('../data/result_log.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(result_log))


    embedding_file_path = '../data/embedding' + city_apply_to + '_' + foundation_model_trained_on + '_SOM_tuned.pth'
    if os.path.exists(embedding_file_path) and load_SpatialInitiate == True:
        embedding_NY = torch.nn.Embedding(num_embeddings=len(loc2index_NY), embedding_dim=embedding_LA.weight.shape[1],
                                          padding_idx=0).to(device)
        embedding_weight = torch.load(embedding_file_path, map_location=device)
        embedding_NY.weight.data=embedding_weight
        som = SOM(embedding_NY.weight.data, gaussian_graph_NY, lr=0.1)
        print("Loaded Spatial Embedding from file.")
        som.embedding_to_map_adj = None
        del gaussian_graph_NY
        gc.collect()
        torch.cuda.empty_cache()
    else:
        with torch.no_grad():
            gaussian_graph_NY = gaussian_graph_NY.double().to(device)
            embedding_NY = embedding_NY.double().to(device)
            embedding_LA = embedding_LA.double().to(device)
            som = SOM(embedding_NY.weight.data, gaussian_graph_NY, lr=0.1)
            som.train(embedding_LA.weight.data, anchor_NY, anchor_NY2LA, anchor_iteration=10,
                      extra_training=0)  # 10 5000+

            som.embedding_to_map[0]=torch.zeros_like(som.embedding_to_map[0])  # set

            som.embedding_to_map_adj = None
            del gaussian_graph_NY
            gc.collect()
            torch.cuda.empty_cache()

            torch.save(som.get_weights(), embedding_file_path)
            print("Saved Spatial Embedding to file.")

    with torch.no_grad():
        weighted = som.get_weights().detach().to(device)
        prediction_model.loc_embedding.weight.data = weighted  # embedding_NY.weight.data.double() torch.nn.Embedding(len(loc2index_NY),embedding_dim=feature_dim).weight.data.double().to(device)   #
        pr_auc, roc_auc = test_foundation_model(prediction_model, test_loader_NY, pooling_method)
        print('SOM tuned Embedding Result: ')
        print(f"PR AUC: {pr_auc:.4f}")
        print(f"AU ROC: {roc_auc:.4f}")

    result_log = load_logfile()
    if additional_information is None:
        result_log[city_apply_to + '_' + foundation_model_trained_on + '_SOM'] = (pr_auc, roc_auc)
    else:
        result_log[
            city_apply_to + '_' + foundation_model_trained_on + '_SOM_' + additional_information] = (
            pr_auc, roc_auc)
    with open('../data/result_log.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(result_log))

    contrastive_dataset = Regression_Dataset_Constractive(test_loader_NY)
    contrastive_dataloader = DataLoader(contrastive_dataset, batch_size=contrastive_batch_size, shuffle=True,
                                        num_workers=16, pin_memory=True, drop_last=True,
                                        collate_fn=contrastive_dataset.collate_fn_pad)

    constractive_save_path = '../data/embedding' + city_apply_to + '_' + foundation_model_trained_on + '_Global_tuned.pth'
    embedding_NY_weight = som.get_weights().clone().detach()
    embedding_NY_weight[0]=torch.zeros_like(embedding_NY_weight[0])
    best_embedding_weight, pr_auc, roc_auc = Global_Tuning(contrastive_dataloader, test_loader_NY,
                                                                     prediction_model, embedding_NY_weight,
                                                                     constractive_save_path,
                                                                     constractive_epochs=contrastive_tuning_epochs,
                                                                     temperature=contrastive_tuning_temp)

    result_log = load_logfile()
    if additional_information is None:
        result_log[city_apply_to + '_' + foundation_model_trained_on + '_GlobalMatching'] = (pr_auc, roc_auc)
    else:
        result_log[
            city_apply_to + '_' + foundation_model_trained_on + '_GlobalMatching_' + additional_information] = (
            pr_auc, roc_auc)
    with open('../data/result_log.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(result_log))

    return pr_auc, roc_auc


if __name__ == '__main__':
    foundation_model_trained_on_list = ['Gowalla_LA','Gowalla_NY','Foursquare_LA','Foursquare_NY',]  #,'Foursquare_LA'
    city_apply_to_list = ['Gowalla_ST'] #'Gowalla_LA','Gowalla_NY','Foursquare_LA','Foursquare_NY'

    for foundation_model_trained_on in foundation_model_trained_on_list:
        for city_apply_to in city_apply_to_list:
            if foundation_model_trained_on == city_apply_to:
                continue
            print(foundation_model_trained_on, city_apply_to)
            #foundation_model_path='../data/Regession_Baseline_NO_Noise_MaxPooling_Foursquare_NY.pth'#'../data/Regession_Baseline_NO_inititate_FoursquareNY.pth'
            #_MSEnoAug_x100_TrajSameWeight
            foundation_model_path='../data/Regession_Baseline_Noise00_MixPooling_'+foundation_model_trained_on+'_MSEnoAug_x100_TrajSameWeight.pth'
            additional_information = 'Noise00_Mix_Pooling_01264_maskedTrajColoc_0.9anchor_som5_AugColocation_2ST'  #'Attention_Pooling'  0.3713 _maskedTrajColoc

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            location_embedding_dim = 256
            test_batch_size = 512
            contrastive_batch_size = 2
            contrastive_tuning_epochs = 2
            contrastive_tuning_temp = 100
            anchor_rate=0.9
            top_n=500
            load_SpatialInitiate = False

            best_pr_auc=0
            for i in range(1,2):
                additional_information+=str(i)
                try:
                    pr_auc, roc_auc=transfer(foundation_model_trained_on,city_apply_to,device,foundation_model_path,feature_dim=location_embedding_dim,test_batch_size=test_batch_size,
                             contrastive_batch_size=contrastive_batch_size,load_SpatialInitiate=load_SpatialInitiate,additional_information=additional_information,
                             contrastive_tuning_epochs=contrastive_tuning_epochs,contrastive_tuning_temp=contrastive_tuning_temp,anchor_rate=anchor_rate, top_n=top_n)

                    if pr_auc>best_pr_auc:
                        best_pr_auc=pr_auc
                        best_roc_auc=roc_auc
                        result_log = load_logfile()
                        result_log[city_apply_to + '_' + foundation_model_trained_on + '_GlobalMatching_' + additional_information] = (pr_auc, roc_auc)
                        with open('../data/result_log.json', 'w', encoding='utf-8') as file:
                            file.write(json.dumps(result_log))

                except Exception as e:
                    print(e)

    # 0.2182