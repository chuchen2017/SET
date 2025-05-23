from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import random

class Regression_Dataset(Dataset):
    def __init__(self, results, loc2index, user_checkin, augmentation=0):  # =loc2index_LA
        self.frind_coloctions = []
        for edge in results:
            if edge == 0:
                continue
            uid1, uid2, frind, colocations, covisited = edge
            traj1 = user_checkin[uid1]
            traj2 = user_checkin[uid2]
            traj1 = [loc2index[loc] for time, loc in traj1 if loc in loc2index.keys()]
            traj2 = [loc2index[loc] for time, loc in traj2 if loc in loc2index.keys()]

            if len(traj1) == 0:
                traj1 = [1]
            if len(traj2) == 0:
                traj2 = [1]

            # if len(traj1) < 1 or len(traj2) < 1:
            #     continue

            # if len(colocations) < 2 and frind == 1: #and frind == 1
            #     continue

            locs = []
            times = []

            covisited_list = []
            covisited_times = []
            for loc, time_stamp in colocations:
                if loc not in loc2index.keys():
                    print(loc)
                    continue
                locs.append(loc2index[loc])
                times.append(time_stamp)

            for loc, time_stamp in covisited:
                if loc not in loc2index.keys():
                    print(loc)
                    continue
                covisited_list.append(loc2index[loc])
                try:
                    year, month, day, hour, minute, second = time_stamp
                except:
                    hour = time_stamp
                # time1 = year*365*24*60+month*30*24*60+day*24*60+hour*60
                covisited_times.append(hour)

            # if len(locs) < 2 :  # 2 1  or len(covisited_list) < 1
            #     continue
            if (len(covisited_list) < 2 or len(locs) < 2):  # frind == 1 and
                continue

            if augmentation != 0 and frind > 0:
                for _ in range(augmentation):
                    sample_index = random.randint(0, len(locs) - 1)
                    locs_sampled = locs[:sample_index] + locs[sample_index + 1:]
                    times_sampled = times[:sample_index] + times[sample_index + 1:]
                    if len(locs_sampled) < 2:
                        continue
                    self.frind_coloctions.append(
                        (locs_sampled, times_sampled, covisited_list, covisited_times, traj1, traj2, frind))
            self.frind_coloctions.append((locs, times, covisited_list, covisited_times, traj1, traj2, frind))

    def __getitem__(self, index):
        return self.frind_coloctions[index][0], self.frind_coloctions[index][1], self.frind_coloctions[index][2], \
        self.frind_coloctions[index][3], \
            self.frind_coloctions[index][4], self.frind_coloctions[index][5], self.frind_coloctions[index][6]

    def __len__(self):
        return len(self.frind_coloctions)

    def collate_fn_pad(self, data):
        x_1_pad = []
        x_2_pad = []
        x_3_pad = []
        x_4_pad = []
        traj1_pad = []
        traj2_pad = []
        y_pad = []

        for x_1, x_2, x_3, x_4, traj1, traj2, y in data:
            x_1_pad.append(torch.tensor(x_1, dtype=torch.long))
            x_2_pad.append(torch.tensor(x_2, dtype=torch.double))
            x_3_pad.append(torch.tensor(x_3, dtype=torch.long))
            x_4_pad.append(torch.tensor(x_4, dtype=torch.long))
            traj1_pad.append(torch.tensor(traj1, dtype=torch.long))
            traj2_pad.append(torch.tensor(traj2, dtype=torch.long))
            y_pad.append(torch.tensor(y, dtype=torch.double))

        x_1_pad = torch.nn.utils.rnn.pad_sequence(x_1_pad, batch_first=True, padding_value=0)
        x_2_pad = torch.nn.utils.rnn.pad_sequence(x_2_pad, batch_first=True, padding_value=0)
        x_3_pad = torch.nn.utils.rnn.pad_sequence(x_3_pad, batch_first=True, padding_value=0)
        x_4_pad = torch.nn.utils.rnn.pad_sequence(x_4_pad, batch_first=True, padding_value=0)
        traj1_pad = torch.nn.utils.rnn.pad_sequence(traj1_pad, batch_first=True, padding_value=0)
        traj2_pad = torch.nn.utils.rnn.pad_sequence(traj2_pad, batch_first=True, padding_value=0)
        y_pad = torch.stack(y_pad).unsqueeze(-1)

        return x_1_pad, x_2_pad, x_3_pad, x_4_pad, traj1_pad, traj2_pad, y_pad

class Regession_Baseline_backup(torch.nn.Module):
    def __init__(self, num_loc, loc_embedding, week_embedding, hour_embedding, feature_dim=128):
        super(Regession_Baseline_backup, self).__init__()

        self.loc_embedding = torch.nn.Embedding(num_embeddings=int(num_loc), embedding_dim=loc_embedding, padding_idx=0)
        self.loc_embedding2 = torch.nn.Embedding(num_embeddings=int(num_loc), embedding_dim=loc_embedding,
                                                 padding_idx=0)
        day_embedding = 0
        self.hour_embedding = torch.nn.Embedding(num_embeddings=26, embedding_dim=loc_embedding, padding_idx=0)
        embedding_size = loc_embedding  # +day_embedding+week_embedding+hour_embedding
        if 1:
            hidden_size = embedding_size

        self.Traj_encoderLSTM = torch.nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=1,
                                              bidirectional=False, batch_first=True)  # 2

        self.Traj_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=embedding_size, nhead=4,
                                             dim_feedforward=hidden_size, dropout=0.1,
                                             activation='relu', batch_first=True), num_layers=1)

        self.Covisited_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=embedding_size, nhead=4,
                                             dim_feedforward=hidden_size, dropout=0.1,
                                             activation='relu', batch_first=True), num_layers=1)
        # self.Covisited_encoder_LSTM=torch.nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=1, bidirectional=False, batch_first=True)

        self.attention = AttentionPooling(embedding_dim=hidden_size)
        self.projection = torch.nn.Linear(hidden_size // 2, 1)  # hidden_size//4  //2

        self.alpha = torch.nn.Linear(hidden_size, hidden_size // 2)  # Parameter(torch.tensor(0.5))
        self.beta = torch.nn.Linear(hidden_size, hidden_size // 2)  # Parameter(torch.tensor(0.5))
        self.weight_traj1 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.weight_traj2 = torch.nn.Linear(hidden_size, hidden_size // 2)

        # self.loc_embedding=self.noisy_loc_embedding

    def forward(self, x, times, covisited, covisited_times, traj1, traj2, training=False):
        # padding
        if training == True:
            His_mask = (x[:, :] == 0)

            x = self.encoder_embedding(x, times, training)

            x = self.Traj_encoder(x, src_key_padding_mask=His_mask)

            #x = torch.mean(x, dim=1, keepdim=False)
            x, _ = torch.max(x, dim=1,keepdim=False)
            # x=self.attention(x)

            lengths1 = (traj1 != 0).sum(dim=1).cpu()
            lengths2 = (traj2 != 0).sum(dim=1).cpu()

            traj1_nonzero_indices = (traj1.sum(dim=1) != 0).nonzero(as_tuple=True)[0]
            traj2_nonzero_indices = (traj2.sum(dim=1) != 0).nonzero(as_tuple=True)[0]
            traj1 = traj1[traj1_nonzero_indices]
            traj2 = traj2[traj2_nonzero_indices]
            lengths1 = lengths1[traj1_nonzero_indices]
            lengths2 = lengths2[traj2_nonzero_indices]

            traj1 = self.noisy_loc_embedding(traj1)
            traj2 = self.noisy_loc_embedding(traj2)

            traj1 = pack_padded_sequence(traj1, lengths1, batch_first=True, enforce_sorted=False)  # padding
            traj1, _ = self.Traj_encoderLSTM(traj1)
            traj1_emb, x_lengths = pad_packed_sequence(traj1, batch_first=True)  # padding

            traj1 = torch.zeros((x.shape[0], traj1_emb.shape[1], traj1_emb.shape[2]), device=traj1_emb.device,
                                dtype=traj1_emb.dtype)
            traj1[traj1_nonzero_indices] = traj1_emb
            traj1 = torch.mean(traj1, dim=1, keepdim=False)

            traj2 = pack_padded_sequence(traj2, lengths2, batch_first=True, enforce_sorted=False)  # padding
            traj2, _ = self.Traj_encoderLSTM(traj2)
            traj2_emb, x_lengths = pad_packed_sequence(traj2, batch_first=True)  # padding

            traj2 = torch.zeros((x.shape[0], traj2_emb.shape[1], traj2_emb.shape[2]), device=traj2_emb.device,
                                dtype=traj2_emb.dtype)
            traj2[traj2_nonzero_indices] = traj2_emb
            traj2 = torch.mean(traj2, dim=1, keepdim=False)

            if covisited.shape[1] > 0:
                covisited_nonzero_indices = (covisited.sum(dim=1) != 0).nonzero(as_tuple=True)[0]
                covisited = covisited[covisited_nonzero_indices]
                covisited_times = covisited_times[covisited_nonzero_indices]
                covisited_emb = self.covisited_embedding(covisited, covisited_times)

                covisited_mask = (covisited[:, :] == 0)
                covisited_emb = self.Covisited_encoder(covisited_emb, src_key_padding_mask=covisited_mask)

                covisited = torch.zeros((x.shape[0], covisited_emb.shape[1], covisited_emb.shape[2]),
                                        device=covisited_emb.device, dtype=covisited_emb.dtype)
                covisited[covisited_nonzero_indices] = covisited_emb

                # covisited=torch.mean(covisited,dim=1,keepdim=False)
                covisited, _ = torch.max(covisited, dim=1, keepdim=False)
                # covisited=self.attention(covisited)

            else:
                covisited = torch.zeros_like(x)

            x = self.alpha(x) + self.beta(covisited) + self.weight_traj1(traj1) + self.weight_traj2(traj2)
            x = self.projection(x)
            return x
        else:
            His_mask = (x[:, :] == 0)
            x = self.encoder_embedding(x, times, training)
            x = self.Traj_encoder(x, src_key_padding_mask=His_mask)

            #x = torch.mean(x, dim=1, keepdim=False)
            #x, _ = torch.max(x, dim=1, keepdim=False)
            x=self.attention(x)

            lengths1 = (traj1 != 0).sum(dim=1).cpu()
            lengths2 = (traj2 != 0).sum(dim=1).cpu()

            # traj1_nonzero_indices = (traj1.sum(dim=1) != 0).nonzero(as_tuple=True)[0]
            # traj2_nonzero_indices = (traj2.sum(dim=1) != 0).nonzero(as_tuple=True)[0]
            # traj1 = traj1[traj1_nonzero_indices]
            # traj2 = traj2[traj2_nonzero_indices]
            # lengths1 = lengths1[traj1_nonzero_indices]
            # lengths2 = lengths2[traj2_nonzero_indices]

            # traj1=self.noisy_loc_embedding(traj1)
            # traj2=self.noisy_loc_embedding(traj2)
            traj1 = self.loc_embedding(traj1)
            traj2 = self.loc_embedding(traj2)

            traj1 = pack_padded_sequence(traj1, lengths1, batch_first=True, enforce_sorted=False)  # padding
            traj1, _ = self.Traj_encoderLSTM(traj1)
            traj1_emb, x_lengths = pad_packed_sequence(traj1, batch_first=True)  # padding

            # traj1 = torch.zeros((x.shape[0], traj1_emb.shape[1], traj1_emb.shape[2]), device=traj1_emb.device,dtype=traj1_emb.dtype)
            # traj1[traj1_nonzero_indices] = traj1_emb
            traj1 = torch.mean(traj1_emb, dim=1, keepdim=False)

            traj2 = pack_padded_sequence(traj2, lengths2, batch_first=True, enforce_sorted=False)  # padding
            traj2, _ = self.Traj_encoderLSTM(traj2)
            traj2_emb, x_lengths = pad_packed_sequence(traj2, batch_first=True)  # padding

            # traj2 = torch.zeros((x.shape[0], traj2_emb.shape[1], traj2_emb.shape[2]), device=traj2_emb.device,dtype=traj2_emb.dtype)
            # traj2[traj2_nonzero_indices] = traj2_emb
            traj2 = torch.mean(traj2_emb, dim=1, keepdim=False)

            if covisited.shape[1] > 0:
                covisited_nonzero_indices = (covisited.sum(dim=1) != 0).nonzero(as_tuple=True)[0]
                covisited = covisited[covisited_nonzero_indices]
                covisited_times = covisited_times[covisited_nonzero_indices]
                covisited_emb = self.covisited_embedding(covisited, covisited_times)

                covisited_mask = (covisited[:, :] == 0)
                covisited_emb = self.Covisited_encoder(covisited_emb, src_key_padding_mask=covisited_mask)

                covisited = torch.zeros((x.shape[0], covisited_emb.shape[1], covisited_emb.shape[2]),
                                        device=covisited_emb.device, dtype=covisited_emb.dtype)
                covisited[covisited_nonzero_indices] = covisited_emb

                #covisited = torch.mean(covisited, dim=1, keepdim=False)
                #covisited, _ = torch.max(covisited, dim=1,keepdim=False)
                covisited=self.attention(covisited)

            else:
                covisited = torch.zeros_like(x)

            x = self.alpha(x) + self.beta(covisited) + self.weight_traj1(traj1) + self.weight_traj2(traj2)
            # x = self.alpha * x + self.beta * covisited
            # x=x+covisited

            x = self.projection(x)
            return x

    def encoder_embedding(self, x, times, training=False):
        if training == False:
            times = torch.mean(times, dim=-1, keepdim=False)
            x_emb = self.loc_embedding(x) * times.unsqueeze(-1)
            return x_emb
        else:
            times = torch.mean(times, dim=-1, keepdim=False)
            x_emb = self.noisy_loc_embedding(x) * times.unsqueeze(-1)
            return x_emb

    def covisited_embedding(self, x, times, training=False):
        # mode with two encoders is better than one!!!
        if training == False:
            x_emb = self.loc_embedding(x) + self.hour_embedding(times)  # *times.unsqueeze(-1)
            return x_emb
        else:
            x_emb = self.noisy_loc_embedding(x) + self.hour_embedding(times)
            return x_emb

    def noisy_loc_embedding(self, x):
        x_emb = self.loc_embedding(x)
        noise = torch.randn_like(x_emb) * 0.001
        x_emb = x_emb + noise
        # x_emb = torch.ones_like(x_emb)
        return x_emb

class Regession_Baseline(torch.nn.Module):
    def __init__(self, num_loc, loc_embedding, week_embedding, hour_embedding, feature_dim=128):
        super(Regession_Baseline, self).__init__()

        self.loc_embedding = torch.nn.Embedding(num_embeddings=int(num_loc), embedding_dim=loc_embedding, padding_idx=0)
        self.loc_embedding2 = torch.nn.Embedding(num_embeddings=int(num_loc), embedding_dim=loc_embedding,
                                                 padding_idx=0)
        day_embedding = 0
        self.hour_embedding = torch.nn.Embedding(num_embeddings=26, embedding_dim=loc_embedding, padding_idx=0)
        embedding_size = loc_embedding  # +day_embedding+week_embedding+hour_embedding
        hidden_size = embedding_size

        self.Traj_encoderLSTM = torch.nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=1,
                                              bidirectional=False, batch_first=True)  # 2

        self.Traj_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=embedding_size, nhead=4,
                                             dim_feedforward=hidden_size, dropout=0.1,
                                             activation='relu', batch_first=True), num_layers=1)

        self.Covisited_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=embedding_size, nhead=4,
                                             dim_feedforward=hidden_size, dropout=0.1,
                                             activation='relu', batch_first=True), num_layers=1)
        self.attention = AttentionPooling(embedding_dim=hidden_size)
        self.projection = torch.nn.Linear(hidden_size // 2, 1)  # hidden_size//4  //2

        self.alpha = torch.nn.Linear(hidden_size, hidden_size // 2)  # Parameter(torch.tensor(0.5))
        self.beta = torch.nn.Linear(hidden_size, hidden_size // 2)  # Parameter(torch.tensor(0.5))
        self.weight_traj1 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.weight_traj2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.pooling_method='Mix'

    def forward(self, x, times, covisited, covisited_times, traj1, traj2, training=False, pooling_method='Mix',contrastive_tuning=False,noise_scale=0.01):
        pooling_method=self.pooling_method
        His_mask = (x[:, :] == 0)

        x = self.encoder_embedding(x, times, training,noise_scale)
        if contrastive_tuning:
            x=torch.ones_like(x)
        x = self.Traj_encoder(x, src_key_padding_mask=His_mask)

        if pooling_method == 'Max':
            x, _ = torch.max(x, dim=1, keepdim=False)
        elif pooling_method == 'Mean':
            x = torch.mean(x, dim=1, keepdim=False)
        elif pooling_method == 'Mix':
            x = torch.mean(x, dim=1, keepdim=False)
        else:
            x = self.attention(x)

        lengths1 = (traj1 != 0).sum(dim=1).cpu()
        lengths2 = (traj2 != 0).sum(dim=1).cpu()

        # traj1 = self.loc_embedding_method(traj1, training,noise_scale)
        # traj2 = self.loc_embedding_method(traj2, training,noise_scale)
        # traj1 = pack_padded_sequence(traj1, lengths1, batch_first=True, enforce_sorted=False)  # padding
        # traj1, _ = self.Traj_encoderLSTM(traj1)
        # traj1, x_lengths = pad_packed_sequence(traj1, batch_first=True)  # padding
        # traj1 = torch.mean(traj1, dim=1, keepdim=False)
        #
        # traj2 = pack_padded_sequence(traj2, lengths2, batch_first=True, enforce_sorted=False)  # padding
        # traj2, _ = self.Traj_encoderLSTM(traj2)
        # traj2, x_lengths = pad_packed_sequence(traj2, batch_first=True)  # padding
        # traj2 = torch.mean(traj2, dim=1, keepdim=False)

        #traj1_nonzero_indices = (traj1.sum(dim=1) != 0).nonzero(as_tuple=True)[0]
        #traj2_nonzero_indices = (traj2.sum(dim=1) != 0).nonzero(as_tuple=True)[0]
        #traj1 = traj1[traj1_nonzero_indices]
        #traj2 = traj2[traj2_nonzero_indices]
        #lengths1 = lengths1[traj1_nonzero_indices]
        #lengths2 = lengths2[traj2_nonzero_indices]
        #print(training,contrastive_tuning,noise_scale)
        traj1=self.loc_embedding_method( traj1, training=training,noise_scale=noise_scale)
        traj2=self.loc_embedding_method( traj2, training=training,noise_scale=noise_scale)
        if contrastive_tuning == True:
            traj1 = torch.ones_like(traj1)
            traj2 = torch.ones_like(traj2)

        traj1 = pack_padded_sequence(traj1, lengths1, batch_first=True, enforce_sorted=False)   # padding
        traj1,_=self.Traj_encoderLSTM(traj1)
        traj1, x_lengths = pad_packed_sequence(traj1, batch_first=True)  # padding
        traj1 = torch.mean(traj1, dim=1, keepdim=False)

        traj2 = pack_padded_sequence(traj2, lengths2, batch_first=True, enforce_sorted=False)   # padding
        traj2,_=self.Traj_encoderLSTM(traj2)
        traj2, x_lengths = pad_packed_sequence(traj2, batch_first=True)  # padding
        traj2 = torch.mean(traj2, dim=1, keepdim=False)

        # traj1 = pack_padded_sequence(traj1, lengths1, batch_first=True, enforce_sorted=False)  # padding
        # traj1, _ = self.Traj_encoderLSTM(traj1)
        # traj1_emb, x_lengths = pad_packed_sequence(traj1, batch_first=True)  # padding
        #
        # traj1 = torch.zeros((x.shape[0], traj1_emb.shape[1], traj1_emb.shape[2]), device=traj1_emb.device,
        #                     dtype=traj1_emb.dtype)
        # traj1[traj1_nonzero_indices] = traj1_emb
        # traj1 = torch.mean(traj1, dim=1, keepdim=False)
        #
        # traj2 = pack_padded_sequence(traj2, lengths2, batch_first=True, enforce_sorted=False)  # padding
        # traj2, _ = self.Traj_encoderLSTM(traj2)
        # traj2_emb, x_lengths = pad_packed_sequence(traj2, batch_first=True)  # padding
        #
        # traj2 = torch.zeros((x.shape[0], traj2_emb.shape[1], traj2_emb.shape[2]), device=traj2_emb.device,
        #                     dtype=traj2_emb.dtype)
        # traj2[traj2_nonzero_indices] = traj2_emb
        # traj2 = torch.mean(traj2, dim=1, keepdim=False)

        covisited_nonzero_indices = (covisited.sum(dim=1) != 0).nonzero(as_tuple=True)[0]
        covisited = covisited[covisited_nonzero_indices]
        covisited_times = covisited_times[covisited_nonzero_indices]
        covisited_emb = self.covisited_embedding(covisited, covisited_times,training,noise_scale)
        covisited_mask = (covisited[:, :] == 0)
        covisited_emb = self.Covisited_encoder(covisited_emb, src_key_padding_mask=covisited_mask)
        covisited = torch.zeros((x.shape[0], covisited_emb.shape[1], covisited_emb.shape[2]),
                                device=covisited_emb.device, dtype=covisited_emb.dtype)
        covisited[covisited_nonzero_indices] = covisited_emb

        if pooling_method == 'Max':
            covisited, _ = torch.max(covisited, dim=1, keepdim=False)
        elif pooling_method == 'Mean':
            covisited = torch.mean(covisited, dim=1, keepdim=False)
        elif pooling_method == 'Mix':
            covisited, _ = torch.max(covisited, dim=1, keepdim=False)
        else:
            covisited = self.attention(covisited)

        x = self.alpha(x) + self.beta(covisited) + self.weight_traj1(traj1) + self.weight_traj1(traj2)

        if contrastive_tuning == True:
            #x = self.projection(x)
            #x = torch.sigmoid(x)
            return x
        else:
            x = self.projection(x)
            #x = torch.sigmoid(x)
            return x

    def encoder_embedding(self, x, times, training=False,noise_scale=0.01):
        times = torch.mean(times, dim=-1, keepdim=False)
        #times = torch.sum(times, dim=-1, keepdim=False)
        x_emb = self.loc_embedding_method(x, training,noise_scale) * times.unsqueeze(-1)
        return x_emb

    def covisited_embedding(self, x, times, training=False,noise_scale=0.01):
        x_emb = self.loc_embedding_method(x, training,noise_scale)  + self.hour_embedding(times)
        return x_emb

    def loc_embedding_method(self, x, training=False,noise_scale=0.01):
        if training == False:
            x_emb = self.loc_embedding(x)
            return x_emb
        else:
            x_emb = self.loc_embedding(x)
            noise = torch.randn_like(x_emb) * noise_scale
            x_emb = x_emb + noise
            return x_emb

class AttentionPooling(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionPooling, self).__init__()
        self.attention = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, embedding_dim)
        weights = self.attention(x)  # (batch_size, seq_len, 1)
        weights = torch.softmax(weights, dim=1)  # (batch_size, seq_len, 1)
        weighted_sum = torch.sum(weights * x, dim=1)  # (batch_size, embedding_dim)
        return weighted_sum
