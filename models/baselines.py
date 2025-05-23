from models.SocialInference import Regession_Baseline
from utils.utils import test_foundation_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

def test_embedding(test_loader_NY,loc2index_LA,embedding_NY,gaussian_graph_NY,foundation_model_path,device,feature_dim=256,trained_generator=None ,adj=None):

    prediction_model = Regession_Baseline(num_loc=len(loc2index_LA) ,loc_embedding=feature_dim ,week_embedding=6
                                          ,hour_embedding=6, feature_dim=feature_dim).double().to(device)
    state_dict = torch.load(foundation_model_path ,map_location=device)
    prediction_model.load_state_dict(state_dict)
    prediction_model.eval()
    if trained_generator is not None:
        source_data = embedding_NY.weight.clone().float()
        source_data = source_data.to(device).double()
        if adj is None:
            source_tensor = trained_generator(source_data.float()).double()
            prediction_model.loc_embedding.weight.data = source_tensor.clone().detach()
        else:
            # adj = adj[:source_data.shape[0],:source_data.shape[0]]
            source_adj = gaussian_graph_NY.to(device)
            source_data_clip = source_data[:source_adj.shape[0]]
            source_tensor = trained_generator(source_data_clip.float() ,source_adj)
            source_data[:source_adj.shape[0]] = source_tensor
            prediction_model.loc_embedding.weight.data = source_data.clone().detach().double()
        pr ,ac = test_foundation_model(prediction_model, test_loader_NY)
    else:
        prediction_model.loc_embedding.weight.data =embedding_NY.weight.clone()
        pr ,ac = test_foundation_model(prediction_model, test_loader_NY)
    return pr ,ac