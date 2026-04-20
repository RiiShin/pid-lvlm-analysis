import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import json
import argparse


batch_size = 256

num_labels_all = 4


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Datasets

class MultimodalDataset(Dataset):
  def __init__(self, data, labels):
    self.data = data
    self.labels = labels
    self.num_modalities = len(self.data)
  
  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return tuple([self.data[i][idx] for i in range(self.num_modalities)] + [self.labels[idx]])



def sinkhorn_probs(matrix, x1_probs, x2_probs):
    matrix = matrix / (torch.sum(matrix, dim=0, keepdim=True) + 1e-8) * x2_probs[None]
    sum = torch.sum(matrix, dim=1)
    if torch.allclose(sum, x1_probs, rtol=0, atol=0.01):
        return matrix, True
    matrix = matrix / (torch.sum(matrix, dim=1, keepdim=True) + 1e-8) * x1_probs[:, None]
    sum = torch.sum(matrix, dim=0)
    if torch.allclose(sum, x2_probs, rtol=0, atol=0.01):
        return matrix, True
    return matrix, False


def mlp(dim, hidden_dim, output_dim, layers, activation):
    activation = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)



class CEAlignment(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, num_labels, layers, activation):
        super().__init__()

        self.num_labels = num_labels
        self.mlp1 = mlp(x1_dim, hidden_dim, embed_dim * num_labels, layers, activation)
        self.mlp2 = mlp(x2_dim, hidden_dim, embed_dim * num_labels, layers, activation)

    def forward(self, x1, x2, x1_probs, x2_probs):
        x1_input = x1
        x2_input = x2

        q_x1 = self.mlp1(x1).unflatten(1, (self.num_labels, -1))
        q_x2 = self.mlp2(x2).unflatten(1, (self.num_labels, -1))

        q_x1 = (q_x1 - torch.mean(q_x1, dim=2, keepdim=True)) / torch.sqrt(torch.var(q_x1, dim=2, keepdim=True) + 1e-8)
        q_x2 = (q_x2 - torch.mean(q_x2, dim=2, keepdim=True)) / torch.sqrt(torch.var(q_x2, dim=2, keepdim=True) + 1e-8)

        # print(q_x1)

        align = torch.einsum('ahx, bhx -> abh', q_x1, q_x2) / math.sqrt(q_x1.size(-1))
        # print(q_x1[0])
        # print(q_x2[0])
        # print(q_x1[0] * q_x2[0])
        # print(torch.sum(q_x1[0] * q_x2[0]))
        # print(align)
        align_logits = align
        align = torch.exp(align)
        # print(x1_input[:10])
        # print(x2_input[:10])
        # print(x1_input[:, 0])
        # print(x2_input[:, 0][None])
        # print(x1_input[:, 0, None] == x2_input[:, 0][None])
        # align = (x1_input[:, 0, None] == x2_input[:, 0][None]) + align - align.detach()

        # print(align[:10, :10])

        normalized = []
        for i in range(align.size(-1)):
            current = align[..., i]
            for j in range(500): # TODO
                current, stop = sinkhorn_probs(current, x1_probs[:, i], x2_probs[:, i])
                if stop:
                    break
            normalized.append(current)
        normalized = torch.stack(normalized, dim=-1)

        if torch.any(torch.isnan(normalized)):
            print(align_logits)
            print(align)
            print(normalized)
            raise Exception('nan')

        return normalized


class CEAlignVLM(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, num_labels,
                 layers, activation, prob_y_x1, prob_y_x2, prob_y_x1x2, p_y, need_softmax=False):
        # x1_dim=product(x1.shape[1:]), x2_dim=product(x2.shape[1:]),
        # hidden_dim=32, embed_dim=10, num_labels=2, layers=3, activation='relu',
        # discrim_1=model_discrim_1, discrim_2=model_discrim_2, discrim_12=model_discrim_12, p_y=p_y
        super().__init__()
        self.num_labels = num_labels
        self.align = CEAlignment(x1_dim, x2_dim, hidden_dim, embed_dim, num_labels, layers, activation)
        self.prob_y_x1 = prob_y_x1
        self.prob_y_x2 = prob_y_x2
        self.prob_y_x1x2 = prob_y_x1x2
        self.need_softmax = need_softmax


        self.register_buffer('p_y', p_y)
        # self.critic_1y = SeparableCritic(x1_dim, y_dim, hidden_dim, embed_dim, layers, activation)
        # self.critic_2y = SeparableCritic(x2_dim, y_dim, hidden_dim, embed_dim, layers, activation)
        # self.critic_12y = SeparableCritic(x1_dim + x2_dim, y_dim, hidden_dim, embed_dim, layers, activation)

    def align_parameters(self):
        return list(self.align.parameters())

    def forward(self, x1, x2, pyx1, pyx2, pyx1x2):
        # print('forward', x1.shape, x2.shape, y.shape)
        with torch.no_grad():
            # a = self.prob_y_x1
            # print('a', a.shape)
            # b = self.prob_y_x2        # basically, use LLM to get the p(y|x1), instead of a simple MLP here.
            # print('b', b.shape)
            p_y_x1 = nn.Softmax(dim=-1)(pyx1)  if self.need_softmax  else pyx1
            p_y_x2 = nn.Softmax(dim=-1)(pyx2)  if self.need_softmax  else pyx2
        align = self.align(torch.flatten(x1, 1, -1), torch.flatten(x2, 1, -1), p_y_x1, p_y_x2)    # anyway, this self.align is MLP and will be updated in the training process. 
        # print(p_y_x2)
        # print(self.p_y)
        # print(y.squeeze(-1))

        self.p_y[self.p_y == 0] += 1e-8
        self.p_y[self.p_y == 1] -= 1e-8

        # sample method: P(X1)
        # coeff: P(Y | X1) Q(X2 | X1, Y)
        # log term: log Q(X2 | X1, Y) - logsum_Y' Q(X2 | X1, Y') Q(Y' | X1)

        q_x2_x1y = align / (torch.sum(align, dim=1, keepdim=True) + 1e-8)
        # print(torch.cat([1 - y, y], dim=-1).shape)
        log_term = torch.log(q_x2_x1y + 1e-8) - torch.log(torch.einsum('aby, ay -> ab', q_x2_x1y, p_y_x1) + 1e-8)[:, :, None]
        # print(q_x2_x1y)
        # print(log_term)
        # That's all we need for optimization purposes
        loss = torch.mean(torch.sum(torch.sum(p_y_x1[:, None, :] * q_x2_x1y * log_term, dim=-1), dim=-1))
        # Now, we calculate the MI terms

        # print(p_y_x2_sampled)
        with torch.no_grad():
            p_y_x1x2 = nn.Softmax(dim=-1)(pyx1x2) if self.need_softmax  else pyx1x2

        p1 = p_y_x1.detach().clone()
        p1[p1 == 0] += 1e-8
        log_p_y_x1 = torch.log(p1)
        # log_p_y_x1[log_p_y_x1 == float("-Inf")] += 1e-8
        p2 = p_y_x2.detach().clone()
        p2[p2 == 0] += 1e-8
        log_p_y_x2 = torch.log(p2)
        # log_p_y_x2[log_p_y_x2 == float("-Inf")] += 1e-8
        p12 = p_y_x1x2.detach().clone()
        p12[p12 == 0] += 1e-8
        log_p_y_x1x2 = torch.log(p12)
        # log_p_y_x1x2[log_p_y_x1x2 == float("-Inf")] += 1e-8

        # mi_y_x1 = torch.mean(torch.log(p_y_x1_sampled) - torch.log(p_y_sampled))
        mi_y_x1 = torch.mean(torch.sum(p_y_x1 * (log_p_y_x1 - torch.log(self.p_y)[None]), dim=-1))
        # mi_y_x2 = torch.mean(torch.log(p_y_x2_sampled) - torch.log(p_y_sampled))
        mi_y_x2 = torch.mean(torch.sum(p_y_x2 * (log_p_y_x2 - torch.log(self.p_y)[None]), dim=-1))
        # mi_y_x1x2 = torch.mean(torch.log(p_y_x1x2_sampled) - torch.log(p_y_sampled))
        mi_y_x1x2 = torch.mean(torch.sum(p_y_x1x2 * (log_p_y_x1x2 - torch.log(self.p_y)[None, None]), dim=-1))
        mi_q_y_x1x2 = p_y_x1[:, None, :] * q_x2_x1y * (log_term + torch.log(p_y_x1 + 1e-8)[:, None, :] - torch.log(self.p_y + 1e-8)[None, None, :])
        '''
        if not self.training:
            print(p_y_x1)
            print(q_x2_x1y)
            print(log_term)
            print(torch.log(p_y_x1))
            print(torch.log(self.p_y))
            print(log_term + torch.log(p_y_x1)[:, None, :] - torch.log(self.p_y)[None, None, :])
        '''
        mi_q_y_x1x2 = torch.sum(torch.sum(mi_q_y_x1x2, dim=-1), dim=-1) # anchored by x1 -- take mean to get MI
        mi_q_y_x1x2 = torch.mean(mi_q_y_x1x2)

        '''
        if not self.training:
            print(torch.stack([mi_y_x1, mi_y_x2, mi_y_x1x2, mi_q_y_x1x2]))
        '''
        # print('   m', torch.stack([mi_y_x1, mi_y_x2, mi_y_x1x2, mi_q_y_x1x2]))

        redundancy = mi_y_x1 + mi_y_x2 - mi_q_y_x1x2
        unique1 = mi_q_y_x1x2 - mi_y_x2
        unique2 = mi_q_y_x1x2 - mi_y_x1
        synergy = mi_y_x1x2 - mi_q_y_x1x2

        # print('   r', torch.stack([redundancy, unique1, unique2, synergy]))

        return loss, torch.stack([redundancy, unique1, unique2, synergy], dim=0), align


def train_ce_alignment(model, train_loader, opt_align, num_epoch=10):
    for _iter in range(num_epoch):
        print(_iter)
        for i_batch, data_batch in enumerate(tqdm(train_loader)):
            opt_align.zero_grad()

            # x1s = [data_batch[]]
            # x2s = [data_batch['vectors2']]
            # ys = [data_batch['labels']]

            # x1s, x2s, ys = [data_batch[0]], [data_batch[1]], [data_batch[-1]]

            # x1_batch = torch.cat(x1s, dim=1).float().cuda()
            # x2_batch = torch.cat(x2s, dim=1).float().cuda()
            # print(x1_batch.shape)
            # print(x1s[0].float().cuda()==x1_batch)
            # print(x2_batch.shape)
            # print(x2s[0].float().cuda()==x2_batch)
            # print(ys[0].shape)
            # y_batch = torch.cat(ys, dim=1).cuda()

            
            # print(len(data_batch))
            # print(data_batch[0].shape)
            # print(data_batch[1].shape)
            # print(data_batch[-1].shape)
            x1_batch, x2_batch, px1_batch, px2_batch, px12_batch, y_batch = [data_batch[i].float().cuda() for i in range(len(data_batch))]
            loss, _, _ = model(x1_batch, x2_batch, px1_batch, px2_batch, px12_batch)
            loss.backward()

            opt_align.step()

            # if (_iter + 1) % 1 == 0 and i_batch % 1 == 0:
            #     print('iter: ', _iter, ' i_batch: ', i_batch, ' align_loss: ', loss.item())
        print(loss.detach().cpu().numpy())


def eval_ce_alignment(model, test_loader):
    results = []
    aligns = []

    for i_batch, data_batch in enumerate(test_loader):
        # x1s = [data_batch['vectors1']]
        # x2s = [data_batch['vectors2']]
        # ys = [data_batch['labels']]

        # x1_batch = torch.cat(x1s, dim=1).float().cuda()
        # x2_batch = torch.cat(x2s, dim=1).float().cuda()
        # y_batch = torch.cat(ys, dim=1).cuda()

        x1_batch, x2_batch, px1_batch, px2_batch, px12_batch, y_batch = [data_batch[i].float().cuda() for i in range(len(data_batch))] 

        with torch.no_grad():
            _, result, align = model(x1_batch, x2_batch, px1_batch, px2_batch, px12_batch)
        results.append(result)
        aligns.append(align)

    results = torch.stack(results, dim=0)
 
    return results, aligns


def critic_ce_alignment(x1, x2, labels, num_labels, train_ds, test_ds, prob_1=None, prob_2=None, prob_12=None, shuffle=True, ce_epochs=10):

    
    p_y = torch.mean(prob_12, dim=0)
    
    # p_y = torch.sum(nn.functional.one_hot(labels.squeeze(-1), num_classes=num_labels), dim=0) / len(labels)
    # print(p_y)

    def product(x):
        return x[0] * product(x[1:]) if x else 1


    model = CEAlignVLM(x1_dim=product(x1.shape[1:]), x2_dim=product(x2.shape[1:]),
        hidden_dim=32, embed_dim=10, num_labels=num_labels, layers=3, activation='relu',
        prob_y_x1=prob_1, prob_y_x2=prob_2, prob_y_x1x2=prob_12,
        p_y=p_y).cuda() # 128, 100, layers=2, relu
    

    opt_align = optim.Adam(model.align_parameters(), lr=1e-3)  # original: lr=1e-3

    train_loader1 = DataLoader(train_ds, shuffle=shuffle, drop_last=True,
                               batch_size=batch_size,
                               num_workers=1)
    test_loader1 = DataLoader(test_ds, shuffle=False, drop_last=True,
                              batch_size=batch_size,
                              num_workers=1)

    # Train and estimate mutual information
    model.train()
    train_ce_alignment(model, train_loader1, opt_align, num_epoch=ce_epochs)

    model.eval()
    results, aligns = eval_ce_alignment(model, test_loader1)
    return results, aligns, (model, prob_1, prob_2, prob_12, p_y)



class VQADataset(Dataset):
    def __init__(self, tensor_list):
        # Initialize dataset with random tensors
        self.vectors1 = tensor_list[0]
        self.vectors2 = tensor_list[1] 
        self.vectors3 = tensor_list[2]
        self.vectors4 = tensor_list[3]
        self.vectors5 = tensor_list[4] 
        self.labels = tensor_list[5]
    
    def __len__(self):
        return self.vectors1.shape[0]
    
    def __getitem__(self, idx):
            # Standard index access
            return (
                self.vectors1[idx],
                self.vectors2[idx],
                self.vectors3[idx],
                self.vectors4[idx],
                self.vectors5[idx],
                self.labels[idx].item()  # Return as Python int
            )


def collect_embeddings_both(json_file_path):
    v_features = []
    l_features = []
    prob_v = []
    prob_l = []
    prob_vl = []
    # New list to store the number of options for each sample
    num_options_list = []


    with open(json_file_path, 'r') as f:
        data = json.load(f)


    for item in data:
        # Store the number of options for the current sample
        num_options_list.append(item.get('num_options'))
        for conv in item['conversations']:
            if conv['from'] == 'gpt':
                # Extract embeddings
                v_features.append(conv['v_feature'])
                l_features.append(conv['l_feature'])
                prob_v.append(conv['v_prob'])
                prob_l.append(conv['l_prob'])
                prob_vl.append(conv['vl_prob'])

    # --- START: New Modification to Standardize Uniform Distributions ---
    # Iterate through all the collected samples to check for uniform failures
    for i in range(len(v_features)):
        num_options = num_options_list[i]

        # We only need to check samples with fewer than 4 options
        if num_options is not None and num_options < 4:
            expected_uniform_val = 1.0 / num_options

            # Check each of the three probability distributions for this sample
            for prob_list in [prob_v, prob_l, prob_vl]:
                current_prob = prob_list[i]
                
                # Heuristic to check if this is a uniform failure distribution
                # It checks if the first 'num_options' values are 1/n and the rest are 0
                is_uniform_failure = True
                # Use np.isclose for safe floating-point comparison
                if not np.allclose(current_prob[:num_options], expected_uniform_val):
                    is_uniform_failure = False
                if not np.allclose(current_prob[num_options:], 0.0):
                    is_uniform_failure = False

                # If it's a uniform failure, replace it with the standard 4-option uniform
                if is_uniform_failure:
                    prob_list[i] = [0.25, 0.25, 0.25, 0.25]
    # --- END: New Modification ---

    v_features = np.array(v_features)
    l_features = np.array(l_features)
    prob_v = np.array(prob_v)
    prob_l = np.array(prob_l)
    prob_vl = np.array(prob_vl)
    
    labels = np.argmax(prob_vl, axis=1)
 

    
    v_features = torch.tensor(v_features, dtype=torch.float32)
    l_features = torch.tensor(l_features, dtype=torch.float32)
    prob_v = torch.tensor(prob_v, dtype=torch.float32)
    prob_l = torch.tensor(prob_l, dtype=torch.float32)
    prob_vl = torch.tensor(prob_vl, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)


    return [v_features, l_features, prob_v, prob_l, prob_vl, labels]


if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Pid estimation")
    parser.add_argument('--directory', type=str, default=None, required=True)
    parser.add_argument('--file_name', type=str, default=None, required=True)

    # Parse arguments
    args = parser.parse_args()

    # Assign parsed arguments to variables
    
    directory = args.directory
    file_name = args.file_name
    
    # directory = './llava/'

    # file_name = '0.5_add.json'

    train_path = directory + 'train/' + file_name
    test_path = directory + 'val/' + file_name

    print('Train path:', train_path)
    print('Test path:', test_path)


    train_tensors = collect_embeddings_both(train_path)

    test_tensors = collect_embeddings_both(test_path)


    train_dataset = VQADataset(train_tensors)
    test_dataset = VQADataset(test_tensors)


    results = critic_ce_alignment(train_dataset.vectors1, train_dataset.vectors2, train_dataset.labels, num_labels_all, 
                        train_dataset, test_dataset, 
                        prob_1=train_dataset.vectors3, prob_2=train_dataset.vectors4, prob_12=train_dataset.vectors5, 
                        shuffle=True, ce_epochs=8) # original # of epochs is 8


    res = results[0].cpu().numpy()
    values = np.mean(res, axis=0)
    values = values/np.log(2)
    values = np.maximum(values, 0)
    print(', '.join([str(v) for v in values]))
    print("Redundancy:", values[0])
    print("Unique1:", values[1])
    print("Unique2:", values[2])
    print("Synergy:", values[3])