import os
import json
import torch
import open_clip
import numpy as np
import datasets
import argparse
from tqdm import tqdm as tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--i', type=str, default='0')
args = parser.parse_args()

"""get cirr testset result, save to json"""
@torch.no_grad()
def test_cirr_submit_result(model, testset, save_dir, batch_size = 16):
    # eval
    model.eval()

    # query feature
    test_queries = testset.test_queries
    all_queries = []
    imgs = []

    pairid = []
    subset = []
    reference_name = []

    visual_query = []
    textual_query = []
    for i, data in enumerate(tqdm(test_queries)):

        visual_query += [data['visual_query']]
        textual_query += [data['textual_query']]
        pairid += [data['pairid']]
        reference_name += [data['reference_name']]
        subset.append(list(data['subset']))
        if len(visual_query) >= batch_size or i == len(test_queries) - 1:
            visual_query = torch.stack(visual_query).float().cuda()

            q = model.extract_query(textual_query, visual_query).data.cpu().numpy()
            all_queries += [q]
            visual_query = []
            textual_query = []
    # all_queries = torch.vstack(all_queries) # (M,D)
    all_queries = np.concatenate(all_queries)

    # targets feature
    candidate_names, candidate_img = testset.test_name_list, testset.test_img_data
    candidate_features = []
    imgs = []
    for i, img_data in enumerate(tqdm(candidate_img)):
        imgs += [img_data]
        if len(imgs) >= batch_size or i == len(candidate_img) - 1:
            if 'torch' not in str(type(imgs[0])):
                imgs = [torch.from_numpy(d).float() for d in imgs]
            imgs = torch.stack(imgs).float().cuda()
            features = model.extract_target(imgs).data.cpu().numpy()
            candidate_features += [features]
            imgs = []
    candidate_features = np.concatenate(candidate_features) # (N,D)

    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(candidate_features.shape[0]):
        candidate_features[i, :] /= np.linalg.norm(candidate_features[i, :])

    sims = - all_queries.dot(candidate_features.T) # (M,N)
    sorted_inds = np.argsort(sims, axis=-1)
    sorted_ind_names = np.array(candidate_names)[sorted_inds] # (M,N)

    mask = torch.tensor(sorted_ind_names != np.repeat(np.array(reference_name), len(candidate_names)).reshape(len(sorted_ind_names),-1)) # (M,N)
    sorted_ind_names = sorted_ind_names[mask].reshape(sorted_ind_names.shape[0], sorted_ind_names.shape[1] - 1) # (M,N-1)

    subset = np.array(subset) # (M,6)
    subset_mask = (sorted_ind_names[..., None] == subset[:, None, :]).sum(-1).astype(bool) # (M,N-1) label elements in subset
    sorted_subset_names = sorted_ind_names[subset_mask].reshape(sorted_ind_names.shape[0], -1) # (M,6)

    pairid_to_gengeral_pred = {str(int(pair_id)): prediction[:50].tolist()  for pair_id, prediction in zip(pairid, sorted_ind_names)}
    pairid_to_subset_pred = {str(int(pair_id)): prediction[:3].tolist() for pair_id, prediction in zip(pairid, sorted_subset_names)}

    general_submission = {'version': 'rc2', 'metric': 'recall'}
    subset_submission = {'version': 'rc2', 'metric': 'recall_subset'}

    general_submission.update(pairid_to_gengeral_pred)
    subset_submission.update(pairid_to_subset_pred)

    print('save cirr test result')
    with open(os.path.join(save_dir, 'test1_pred_ranks_recall_{}.json'.format(args.i)), 'w+') as f:
        json.dump(general_submission, f, sort_keys=True)
        
    with open(os.path.join(save_dir, 'test1_pred_ranks_recall_subset_{}.json'.format(args.i)), 'w+') as f:
        json.dump(subset_submission, f, sort_keys=True)

if __name__ == '__main__':
    clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2B-s32B-b79K')
    testset = datasets.CIRR(path='../data/CIRR',transform=[preprocess_train, preprocess_val])
    model_dir = './checkpoints/'
    # load model
    model = torch.load(os.path.join(model_dir, 'cirr_{}_best_model.pt'.format(args.i)))
    # generate submission json file
    test_cirr_submit_result(model, save_dir=model_dir, testset=testset, batch_size=16)
