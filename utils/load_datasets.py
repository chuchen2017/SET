import torch
import json

def load_dataset(dataset_city):
    if 'Gowalla' in dataset_city:
        if 'NY' in dataset_city:
            checkin_path='../data/Gowalla_uid_time_checkinNY.json'
            colocation_path='../data/Gowalla_resultsNY_filtered.json'
        elif 'LA' in dataset_city:
            checkin_path='../data/Gowalla_uid_time_checkinLA.json'
            colocation_path='../data/Gowalla_resultsLA_filtered.json'
        else:
            checkin_path='../data/Gowalla_uid_time_checkinST.json'
            colocation_path='../data/Gowalla_resultsST_filtered.json'
    else:
        if 'NY' in dataset_city:
            checkin_path='../data/Foursquare_uid_time_checkinNY.json'
            colocation_path='../data/Foursquare_resultsNY_filtered.json'
        else:
            checkin_path='../data/Foursquare_uid_time_checkinLA.json'
            colocation_path='../data/Foursquare_resultsLA_filtered.json'

    uid_time_checkin = json.loads(open(checkin_path).read())
    colocation = json.loads(open(colocation_path).read())
    visited = []
    for edge in colocation:
        uid1,uid2,frind,colocations,covisited = edge
        visited+=[loc for loc,time in colocations]
    loc2index = {loc:i for i,loc in enumerate(set(visited))}
    return uid_time_checkin,colocation,loc2index


