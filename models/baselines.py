from models.SocialInference import Regession_Baseline
from utils.utils import test_foundation_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the generator network (maps embeddings A to B)
class Generator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Generator, self).__init__()
        self.model = MLP(embedding_dim, hidden_dim, embedding_dim)

    def forward(self, x):
        return self.model(x)

# Define the discriminator network (critic for the embeddings)
class Discriminator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = MLP(embedding_dim, hidden_dim, 1)

    def forward(self, x):
        return self.model(x)

# Gradient penalty calculation
def gradient_penalty(D, real_data, fake_data, device):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, device=device).expand_as(real_data)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty

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