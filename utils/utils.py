import torch
import math
import numpy as np
from sklearn.metrics import roc_auc_score,precision_recall_curve, auc
from tqdm import tqdm

def gaussian_kernel(adj, sigma=1.0):
    # Compute the Gaussian kernel
    adj = 1 - torch.exp(-adj ** 2 / (2 * sigma ** 2))
    return adj

# def weighted_sampling(location_entropy, num_anchor_points):
#     entropy_values = np.array(list(location_entropy.values()))
#     probabilities = entropy_values / entropy_values.sum()
#     sampled_indices = np.random.choice(len(location_entropy), size=num_anchor_points, replace=False,p=probabilities)
#     sampled_keys = [list(location_entropy.keys())[i] for i in sampled_indices]
#     return sampled_keys

def weighted_sampling(location_entropy, num_anchor_points):
    anchor_points = np.linspace(0, len(location_entropy), num_anchor_points + 1)
    anchor_points = [int(i) for i in anchor_points][:num_anchor_points]
    return [list(location_entropy.keys())[i] for i in anchor_points]

def graph_construction(loc2index,uid_time_checkin,colocation=None):
    graph_LA = torch.zeros((len(loc2index), len(loc2index)))
    for uid, checkin in uid_time_checkin.items():
        checkin = [loc2index[loc] for time, loc in checkin if loc in loc2index]
        for i in range(1, len(checkin)):
            graph_LA[checkin[i - 1], checkin[i]] += 1
            graph_LA[checkin[i], checkin[i - 1]] += 1

    if colocation is not None:
        for edge in colocation:
            if edge == 0:
                continue
            uid1,uid2,frind,colocations,covisited = edge
            for i in range(1, len(colocations)):
                loc1_index=loc2index[colocations[i - 1][0]]
                loc2_index=loc2index[colocations[i][0]]
                graph_LA[loc1_index, loc2_index] += 1
                graph_LA[loc2_index, loc1_index] += 1

    gaussian_graph_LA = gaussian_kernel(graph_LA)
    gaussian_graph_LA += torch.eye(len(gaussian_graph_LA))
    return gaussian_graph_LA

def location_entropy_construction(loc2index,uid_time_checkin,colocation=None):
    location_visiting_num = {}
    for uid, checkins in uid_time_checkin.items():
        for time, loc in checkins:
            if loc not in loc2index:
                continue
            visit_time = len([time for time, loc in checkins if loc == loc])
            location_visiting_num.setdefault(loc2index[loc], []).append(visit_time)

    if colocation is not None:
        for edge in colocation:
            uid1, uid2, frind, colocations, covisited = edge
            visited = [loc2index[loc] for loc, time in colocations]
            for loc in visited:
                location_visiting_num.setdefault(loc, []).append(1)

    location_entropy_LA = {}
    for loc, visit_times in location_visiting_num.items():
        total_visits = sum(visit_times)  # Total visits for the location
        if total_visits == 0:
            continue
        probabilities = [vt / total_visits for vt in visit_times]
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        location_entropy_LA[loc] = entropy
    location_entropy_LA = {k: v for k, v in sorted(location_entropy_LA.items(), key=lambda item: item[1], reverse=True)}
    return location_entropy_LA

def sample_anchors(location_entropy_LA,location_entropy_NY,num_anchor_points = 8000, top_n = 200):
    anchor_LA = weighted_sampling(dict(list(location_entropy_LA.items())[top_n + 1:]), num_anchor_points)
    anchor_NY = weighted_sampling(dict(list(location_entropy_NY.items())[top_n + 1:]), num_anchor_points)
    anchor_LA_added = list(location_entropy_LA.keys())[:top_n]
    anchor_NY_added = list(location_entropy_NY.keys())[:top_n]
    anchor_LA2NY = dict(zip(anchor_LA_added, anchor_NY_added))
    anchor_NY2LA = dict(zip(anchor_NY_added, anchor_LA_added))
    for loc_id in range(len(anchor_LA)):
        loc = anchor_LA[loc_id]
        if loc in anchor_LA2NY:
            continue
        anchor_LA2NY[loc] = anchor_NY[loc_id]
        anchor_NY2LA[anchor_NY[loc_id]] = loc
    anchor_LA = list(anchor_LA2NY.keys())
    anchor_NY = list(anchor_NY2LA.keys())
    return anchor_LA, anchor_NY

def test_foundation_model(prediction_model,test_loader,pooling_method='Mix'):
    with torch.no_grad():
        device = next(prediction_model.parameters()).device
        prediction_model.eval()
        test_bar = tqdm(test_loader)
        predictions = []
        labels = []
        for x,times,covisited,covisited_time,traj1,traj2, y in test_bar:
            x, times,covisited,covisited_time,traj1,traj2, y = x.to(device),times.to(device),covisited.to(device),covisited_time.to(device),traj1.to(device),traj2.to(device), y.to(device)
            prediction = prediction_model(x,times,covisited,covisited_time,traj1,traj2,training=False,pooling_method=pooling_method,contrastive_tuning=False)
            predictions.append(prediction.detach().cpu().view(-1).tolist())
            labels.append(y.detach().cpu().view(-1).tolist())
        labels = sum(labels, [])
        predictions = sum(predictions, [])
        #clasification = [1 if pred > 50 else 0 for pred in predictions]
        #print(sum(clasification)/len(clasification),(sum(y)/len(y)).item())
        roc_auc = roc_auc_score(labels, predictions)
        precision, recall, _ = precision_recall_curve(labels, predictions)
        pr_auc = auc(recall, precision)
        # print(f"PR AUC: {pr_auc:.4f}")
        # print(f"AU ROC: {roc_auc:.4f}")
    return pr_auc,roc_auc
