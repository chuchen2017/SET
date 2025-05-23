import torch
from torch.cuda import device
from torch.utils.data import Dataset
import random

from torch_geometric.graphgym import train
from tqdm import tqdm
from utils.utils import test_foundation_model
import torch.optim as optim

class Regression_Dataset_Constractive(Dataset):
    def __init__(self, test_dataloader):  # =loc2index_LA
        self.frind_coloctions = []
        for x, times, covisited, covisited_time, traj1, traj2, y in test_dataloader:
            for batch in range(x.shape[0]):
                if len(set(covisited[batch].tolist())) < 2 or len(set(x[batch].tolist())) < 2:
                    continue
                self.frind_coloctions.append((x[batch].clone(), times[batch].clone(), covisited[batch].clone(),
                                              covisited_time[batch].clone(), traj1[batch].clone(), traj2[batch].clone(),
                                              y[batch].clone()))
        self.contrastive_pairs = []

        for index in range(len(self.frind_coloctions)):
            x = self.frind_coloctions[index][0]
            times = self.frind_coloctions[index][1]
            covisited = self.frind_coloctions[index][2]
            covisited_time = self.frind_coloctions[index][3]
            traj1 = self.frind_coloctions[index][4]
            traj2 = self.frind_coloctions[index][5]

            contrastive_pairs_user = []
            covisited_locations = set(covisited.tolist())
            while len(covisited_locations) >= 2:
                loc1, loc2 = random.sample(list(covisited_locations), k=2)
                covisited_locations.remove(loc1)
                covisited_locations.remove(loc2)

                x1 = x[x != loc1]
                times1 = times[x != loc1]
                covisited1 = covisited[covisited != loc1]
                covisited_time1 = covisited_time[covisited != loc1]
                traj11 = traj1[traj1 != loc1]
                traj21 = traj2[traj2 != loc1]
                pair1 = (x1, times1, covisited1, covisited_time1, traj11, traj21, self.frind_coloctions[index][6])

                x2 = x[x != loc2]
                times2 = times[x != loc2]
                covisited2 = covisited[covisited != loc2]
                covisited_time2 = covisited_time[covisited != loc2]
                traj12 = traj1[traj1 != loc2]
                traj22 = traj2[traj2 != loc2]
                pair2 = (x2, times2, covisited2, covisited_time2, traj12, traj22, self.frind_coloctions[index][6])

                if covisited1.shape[0] == 0 or covisited2.shape[0] == 0 or x1.shape[0] == 0 or x2.shape[0] == 0:
                    break
                contrastive_pairs_user.append((pair1, pair2))
            # x_set = set(x.tolist())
            # while len(x_set) >= 4:
            #     loc1, loc2 = random.sample(list(x_set), k=2)
            #     x_set.remove(loc1)
            #     x_set.remove(loc2)
            #
            #     x1 = x[x != loc1]
            #     times1 = times[x != loc1]
            #     covisited1 = covisited[covisited != loc1]
            #     covisited_time1 = covisited_time[covisited != loc1]
            #     traj11 = traj1[traj1 != loc1]
            #     traj21 = traj2[traj2 != loc1]
            #     pair1 = (x1, times1, covisited1, covisited_time1, traj11, traj21, self.frind_coloctions[index][6])
            #
            #     x2 = x[x != loc2]
            #     times2 = times[x != loc2]
            #     covisited2 = covisited[covisited != loc2]
            #     covisited_time2 = covisited_time[covisited != loc2]
            #     traj12 = traj1[traj1 != loc2]
            #     traj22 = traj2[traj2 != loc2]
            #     pair2 = (x2, times2, covisited2, covisited_time2, traj12, traj22, self.frind_coloctions[index][6])
            #
            #     contrastive_pairs_user.append((pair1, pair2))

            if len(contrastive_pairs_user) == 0:
                continue
            self.contrastive_pairs.append(contrastive_pairs_user)

    def __getitem__(self, index):
        index1 = random.sample(range(len(self.contrastive_pairs[index])), k=1)[0]
        return self.contrastive_pairs[index][index1]

    def __len__(self):
        return len(self.contrastive_pairs)

    def collate_fn_pad(self, data):

        x_1_pad1 = []
        x_2_pad1 = []
        x_3_pad1 = []
        x_4_pad1 = []
        traj1_pad1 = []
        traj2_pad1 = []
        y_pad1 = []

        x_1_pad2 = []
        x_2_pad2 = []
        x_3_pad2 = []
        x_4_pad2 = []
        traj1_pad2 = []
        traj2_pad2 = []
        y_pad2 = []

        for pair1, pair2 in data:
            x_1, x_2, x_3, x_4, traj1, traj2, y = pair1
            x_1_2, x_2_2, x_3_2, x_4_2, traj1_2, traj2_2, y_2 = pair2

            x_1_pad1.append(x_1)
            x_2_pad1.append(x_2)
            x_3_pad1.append(x_3)
            x_4_pad1.append(x_4)
            traj1_pad1.append(traj1)
            traj2_pad1.append(traj2)
            y_pad1.append(y)

            x_1_pad2.append(x_1_2)
            x_2_pad2.append(x_2_2)
            x_3_pad2.append(x_3_2)
            x_4_pad2.append(x_4_2)
            traj1_pad2.append(traj1_2)
            traj2_pad2.append(traj2_2)
            y_pad2.append(y_2)

        x_1_pad1 = torch.nn.utils.rnn.pad_sequence(x_1_pad1, batch_first=True, padding_value=0)
        x_2_pad1 = torch.nn.utils.rnn.pad_sequence(x_2_pad1, batch_first=True, padding_value=0)
        x_3_pad1 = torch.nn.utils.rnn.pad_sequence(x_3_pad1, batch_first=True, padding_value=0)
        x_4_pad1 = torch.nn.utils.rnn.pad_sequence(x_4_pad1, batch_first=True, padding_value=0)
        traj1_pad1 = torch.nn.utils.rnn.pad_sequence(traj1_pad1, batch_first=True, padding_value=0)
        traj2_pad1 = torch.nn.utils.rnn.pad_sequence(traj2_pad1, batch_first=True, padding_value=0)
        y_pad1 = torch.tensor(y_pad1, dtype=torch.double)

        x_1_pad2 = torch.nn.utils.rnn.pad_sequence(x_1_pad2, batch_first=True, padding_value=0)
        x_2_pad2 = torch.nn.utils.rnn.pad_sequence(x_2_pad2, batch_first=True, padding_value=0)
        x_3_pad2 = torch.nn.utils.rnn.pad_sequence(x_3_pad2, batch_first=True, padding_value=0)
        x_4_pad2 = torch.nn.utils.rnn.pad_sequence(x_4_pad2, batch_first=True, padding_value=0)
        traj1_pad2 = torch.nn.utils.rnn.pad_sequence(traj1_pad2, batch_first=True, padding_value=0)
        traj2_pad2 = torch.nn.utils.rnn.pad_sequence(traj2_pad2, batch_first=True, padding_value=0)
        y_pad2 = torch.tensor(y_pad2, dtype=torch.double)

        pair1 = (x_1_pad1, x_2_pad1, x_3_pad1, x_4_pad1, traj1_pad1, traj2_pad1, y_pad1)
        pair2 = (x_1_pad2, x_2_pad2, x_3_pad2, x_4_pad2, traj1_pad2, traj2_pad2, y_pad2)

        return pair1, pair2

def Contrastive_Tuning(temperature, model, data_loader, train_optimizer, prediction_model, epoch, epochs):
    model.train()
    prediction_model.train()
    device = next(model.parameters()).device
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pair1, pair2 in train_bar:
        prediction_model.loc_embedding = model  # .weight.data#.clone().detach()
        torch.cuda.empty_cache()
        x, times, covisited, covisited_time, traj1, traj2, y = pair1
        x_2, times_2, covisited_2, covisited_time_2, traj1_2, traj2_2, y_2 = pair2

        x, times, covisited, covisited_time, traj1, traj2, y = x.to(device), times.to(device), covisited.to(
            device), covisited_time.to(device), traj1.to(device), traj2.to(device), y.to(device)
        x_2, times_2, covisited_2, covisited_time_2, traj1_2, traj2_2, y_2 = x_2.to(device), times_2.to(
            device), covisited_2.to(device), covisited_time_2.to(device), traj1_2.to(device), traj2_2.to(
            device), y_2.to(device)

        out_1 = prediction_model(x, times, covisited, covisited_time, traj1, traj2, training=False, pooling_method='Mix',contrastive_tuning=True,noise_scale=0.0)
        out_2 = prediction_model(x_2, times_2, covisited_2, covisited_time_2, traj1_2, traj2_2, training=False, pooling_method='Mix',contrastive_tuning=True,noise_scale=0.0)

        batch_size = out_1.shape[0]
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous())/  temperature)  #torch.norm(out) /

        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        # print('similarity ',sim_matrix)
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1)/   temperature)  #torch.norm(out_1) /
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        # print('pos_sim ',pos_sim)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        # loss.requires_grad = True
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return model,prediction_model

def Global_Tuning(contrastive_dataloader,test_loader_NY,prediction_model,embedding_NY_weight,constractive_save_path,constractive_epochs = 5,temperature = 100,pooling_method='Mix' ):
    device = next(prediction_model.parameters()).device

    prediction_model = prediction_model.to(device)
    model = torch.nn.Embedding(num_embeddings=embedding_NY_weight.shape[0], embedding_dim=embedding_NY_weight.shape[1], padding_idx=0).double().to(device)  # , padding_idx=0
    model.weight.data = embedding_NY_weight.clone().detach().to(device)

    # model.weight.data = torch.randn_like(model.weight.data)
    model.weight.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # , weight_decay=1e-6, weight_decay=1e-6  , weight_decay=1e-6  lr=1e-3 the best
    # froze the parameters of prediction_model
    for param in prediction_model.parameters():
        param.requires_grad = False

    prediction_model.loc_embedding = model
    prediction_model.loc_embedding.weight.requires_grad = True

    best_result = 0.0
    best_roc_auc = 0.0

    epoch=1
    train_or_not = True
    while epoch <= constractive_epochs and train_or_not:
        torch.cuda.empty_cache()
        #model,prediction_model = Contrastive_Tuning(temperature,model, contrastive_dataloader, optimizer, prediction_model, epoch=epoch,epochs=constractive_epochs) #train_loss =
        Contrastive_Tuning(temperature, model, contrastive_dataloader, optimizer,prediction_model, epoch=epoch,epochs=constractive_epochs)  # train_loss =
        epoch+=1
        if epoch % 1 == 0:
            #prediction_model.loc_embedding = model
            prediction_model.loc_embedding.weight.data = model.weight.data.clone().detach()
            pr_auc, roc_auc = test_foundation_model(prediction_model, test_loader_NY,pooling_method)
            print('Global Tuning Embedding Result: ',epoch)
            print(f"PR AUC: {pr_auc:.4f}")
            print(f"AU ROC: {roc_auc:.4f}")
            train_or_not = False
            if pr_auc > best_result:
                train_or_not = True
                best_result = pr_auc
                best_roc_auc = roc_auc
                best_embedding_weight = model.weight.data.clone().detach()
                if constractive_save_path is not None:
                    torch.save(best_embedding_weight, constractive_save_path)

    return best_embedding_weight,best_result,best_roc_auc