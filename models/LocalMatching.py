import torch
from tqdm import tqdm

class SOM(torch.nn.Module):
    def __init__(self, embedding_to_map, embedding_to_map_adj, lr=0.1):
        self.embedding_to_map = embedding_to_map
        self.embedding_to_map_adj = embedding_to_map_adj
        self.lr = lr

    def update_weights(self, x,  iter, idx, max_iter=None):
        # if max_iter is not None:
        #     lr = self.lr * (1 - iter / max_iter)
        # else:
        #     lr = self.lr * (1 - iter / self.max_iter)

        # x = x.repeat(self.embedding_to_map.shape[0], 1)
        # neighbors = self.embedding_to_map_adj[idx] > 0
        # neighbors = neighbors.nonzero(as_tuple=True)[0]#.tolist()#[0]
        # lr = torch.ones_like(x) * lr
        # self.embedding_to_map[neighbors] += lr[neighbors] * self.embedding_to_map_adj[neighbors,idx].unsqueeze(1) * (x[neighbors] - self.embedding_to_map[neighbors])# *0.5

        if max_iter is not None:
            lr = self.lr * (1 - iter / max_iter)
        else:
            lr = self.lr * (1 - iter / self.max_iter)
        direction = x - self.embedding_to_map[idx]
        self.embedding_to_map[idx] += lr * direction
        # Update the weights of the BMU's neighbors
        neighbors = self.embedding_to_map_adj[idx] > 0
        neighbors = neighbors.nonzero(as_tuple=True)[0]  # .tolist()#[0]

        for neighbor in neighbors:
            if neighbor == idx:
                continue
            self.embedding_to_map[neighbor] += lr * self.embedding_to_map_adj[idx, neighbor] * (x - self.embedding_to_map[neighbor])  # *0.5

    def train(self, embedding_map_to, anchor_NY, anchor_NY2LA, anchor_iteration=2, extra_training=1000):
        train_bar = tqdm(range(anchor_iteration))
        for i in train_bar:
            #print('anchor_iteration ', i)
            for anchor in range(len(anchor_NY)):
                idx = anchor_NY[anchor]
                #x = self.embedding_to_map[idx].squeeze()
                x_index = torch.argmax(torch.cosine_similarity(self.embedding_to_map[idx].unsqueeze(0), embedding_map_to, dim=1))
                x = embedding_map_to[x_index]

                #x = embedding_map_to[anchor_NY2LA[idx]]
                self.update_weights(x,  i * len(anchor_NY) + anchor, idx, anchor_iteration * len(anchor_NY))

        for iter in range(anchor * anchor_iteration, anchor * anchor_iteration + extra_training):
            idx = torch.randint(0, self.embedding_to_map.shape[0], (1,), device=self.embedding_to_map.device)  # Randomly select a data point
            x_index = torch.argmax(torch.cosine_similarity(self.embedding_to_map[idx].unsqueeze(0), embedding_map_to, dim=1))
            x = embedding_map_to[x_index]

            # nearest_neighbour = torch.argmax(torch.cosine_similarity(self.embedding_to_map[idx].squeeze(), embedding_map_to))
            # x = embedding_map_to[nearest_neighbour]
            #nearest_neighbour = torch.argmax(torch.cosine_similarity(x, embedding_map_to))
            self.update_weights(x, iter, idx, max_iter=anchor_iteration * len(anchor_NY) + extra_training)

    def get_weights(self):
        """
        Get the trained weights of the SOM.
        """
        return self.embedding_to_map


def Spatial_Initiate(embedding_LA,loc2index_LA,gaussian_graph_LA,anchor_LA,loc2index_NY,gaussian_graph_NY,anchor_NY,anchor_NY2LA,device,noise_level = 0.01,top_k = 5):
    with torch.no_grad():
        #i=0
        embedding_NY = torch.nn.Embedding(num_embeddings=len(loc2index_NY), embedding_dim=embedding_LA.weight.shape[1],padding_idx=0).to(device)

        for anchor in anchor_NY:
            embedding_NY.weight.data[anchor] = embedding_LA.weight.data[anchor_NY2LA[anchor]]  # + noise

        location_points=set(list(range(len(loc2index_NY))))-set(anchor_NY)
        location_points_bar=tqdm(list(location_points))
        for loc in location_points_bar:
            noise = torch.randn_like(embedding_LA.weight.data[0]) * noise_level
            connection_strength = gaussian_graph_NY[loc, anchor_NY]
            if sum(connection_strength) == 0:
                embedding_NY.weight.data[loc] = embedding_LA.weight.data[torch.randint(0, len(loc2index_LA), (1,))] + noise
            else:
                strongest_connection = anchor_NY[torch.argmax(connection_strength)]
                embedding_NY.weight.data[loc] = embedding_LA.weight.data[anchor_NY2LA[strongest_connection]] + noise


        # train_bar = tqdm(range(embedding_NY.weight.data.shape[0]))
        # for loc in train_bar:
        #     if loc in anchor_NY:
        #         #connection_strength = gaussian_graph_NY[loc, anchor_NY]
        #         # if sum(connection_strength) == 0:
        #         #     embedding_NY.weight.data[loc] = embedding_LA.weight.data[torch.randint(0, len(loc2index_LA), (1,))] + noise
        #         # else:
        #         #     k_strongest_connection = torch.topk(connection_strength, top_k).indices
        #         #     k_strongest_connection = [anchor_NY2LA[anchor_NY[k]] for k in k_strongest_connection]
        #         #     embedding_NY.weight.data[loc] = torch.mean(embedding_LA.weight.data[k_strongest_connection],dim=0)  # +noise
        #         embedding_NY.weight.data[loc] = embedding_LA.weight.data[anchor_NY2LA[loc]]# + noise
        #     else:
        #         connection_strength = gaussian_graph_NY[loc, anchor_NY]
        #         if sum(connection_strength) == 0:
        #             i+=1
        #             embedding_NY.weight.data[loc] = embedding_LA.weight.data[torch.randint(0, len(loc2index_LA), (1,))] + noise
        #         else:
        #             strongest_connection = anchor_NY[torch.argmax(connection_strength)]
        #             embedding_NY.weight.data[loc] = embedding_LA.weight.data[anchor_NY2LA[strongest_connection]] + noise
    #print(i,i/len(loc2index_NY))
    return embedding_NY