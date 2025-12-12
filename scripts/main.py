import dill
import sys
import os

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_directory, '../'))
sys.path.append(os.path.join(current_directory, './gnn'))
sys.path.append(os.path.join(current_directory, '../datasets'))

import time
import numpy as np
from tqdm import tqdm
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader as torch_DataLoader
import pandas as pd
from torch_geometric.loader import DataLoader as pyg_DataLoader
from copy import deepcopy
from common_layers import *
from gnn import GNNGraph
from gnn import graph_batch_from_smile
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params
from collections import defaultdict
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import multilabel_margin_loss
import math
import wandb
from datasets import (
    MIMIC_III_Datasets_text_final,  MIMIC_III_Datasets_text_Eval_final, MIMIC_III_Datasets_text_Test_final, MIMIC_III_Datasets_text_Test_final_my,
    MIMIC_IV_Datasets_text_final,  MIMIC_IV_Datasets_text_Eval_final, MIMIC_IV_Datasets_text_Test_final, MIMIC_IV_Datasets_text_Test_final_my,
)


def save_model(save_best, epoch=None, model_name=""):
    if args.output_model_dir is not None:
        if model_name != "":
            model_file = "model_{}.pth".format(model_name)

        epoch = str(epoch)
        if not os.path.exists(os.path.join(args.output_model_dir, epoch)):
            os.makedirs(os.path.join(
                os.path.join(args.output_model_dir, epoch)))

        text2latent_saved_file_path = os.path.join(
            args.output_model_dir, epoch, "text2latent_{}".format(model_file))
        torch.save(text2latent.state_dict(), text2latent_saved_file_path)

        mol2latent_saved_file_path = os.path.join(
            args.output_model_dir, epoch, "mol2latent_{}".format(model_file))
        torch.save(mol2latent.state_dict(), mol2latent_saved_file_path)

        aggregator_saved_file_path = os.path.join(
            args.output_model_dir, epoch, "aggregator_{}".format(model_file))
        torch.save(aggregator.state_dict(), aggregator_saved_file_path)

        dsp_aggregator_saved_file_path = os.path.join(
            args.output_model_dir, epoch, "dsp_aggregator_{}".format(model_file))
        torch.save(dsp_aggregator.state_dict(), dsp_aggregator_saved_file_path)

        proj_sdp_saved_file_path = os.path.join(
            args.output_model_dir, epoch, "proj_dsp_{}".format(model_file))
        torch.save(proj_dsp.state_dict(), proj_sdp_saved_file_path)

        global_encoder_saved_file_path = os.path.join(
            args.output_model_dir, epoch, "global_encoder_{}".format(model_file))
        torch.save(global_encoder.state_dict(), global_encoder_saved_file_path)

    return

def build_atc3_mapping(med_voc, device):
    atc3_to_idx = {}
    mapping = []
    num_atc3 = 0
    # med_voc.idx2word is usually a dict {index: word}
    for i in range(len(med_voc.idx2word)):
        code = med_voc.idx2word[i]
        # Assuming code is string. If it's ATC4 code (5 chars), take 4.
        atc3 = code[:4]
        if atc3 not in atc3_to_idx:
            atc3_to_idx[atc3] = num_atc3
            num_atc3 += 1
        mapping.append(atc3_to_idx[atc3])
    return torch.tensor(mapping, device=device), num_atc3

@torch.no_grad()
def eval_epoch(dataloader, batch_size, med_vocab_size, input_med_rep):
    criterion = binary_cross_entropy_with_logits
    text2latent.eval()
    mol2latent.eval()
    aggregator.eval()
    global_encoder.eval()

    dsp_aggregator.eval()
    proj_dsp.eval()

    accum_loss_multi, accum_loss_bce, accum_loss_rec = 0, 0, 0
    smm_record, ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(6)]
    visit_cnt = 0
    med_cnt = 0

    print("Start testing!")
    start_time = time.time()
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    smm_record = []
    for step, batch in enumerate(L):
        y_gt, y_gt_his, y_pred, y_pred_prob, y_pred_label = [[] for _ in range(5)]
        batch_num = batch[0].size(0)

        global_embeddings = global_encoder(**MPNN_drug_data)
        global_embeddings = torch.mm(average_projection, global_embeddings)

        records_dps_txt = batch[0].to(device)
        records_med_txt = batch[1].to(device)
        med_index = batch[2]
        records_dia_txt = batch[3].to(device)
        records_pro_txt = batch[4].to(device)
        records_sym_txt = batch[5].to(device)
        bce_target = batch[6].to(device)
        multi_target = batch[7].to(device)

        his_med_embeddings = batch[8].to(device)
        his_mask = batch[9].to(device)
        
        med_index = [list(map(int,  elem.split(" "))) for elem in med_index]
        
        description_repr = records_dps_txt

        description_repr_input = description_repr

        attn_dsp = dsp_aggregator(description_repr_input, torch.cat([torch.unsqueeze(records_sym_txt, 1), torch.unsqueeze(
            records_dia_txt, 1), torch.unsqueeze(records_pro_txt, 1)], dim=1), mask=None)

        description_repr = torch.cat(
            [description_repr_input, attn_dsp], dim=-1)

        description_repr = proj_dsp(description_repr)

        his_med_repr, _ = aggregator(
            description_repr, his_med_embeddings, mask=his_mask)

        description_repr = text2latent(description_repr)

        # GNN+LLM => encode medications
        med_rep = torch.cat([input_med_rep, global_embeddings], dim=1)

        med_rep = mol2latent(med_rep)

        description_repr = description_repr + his_med_repr
        
        output = torch.mm(description_repr, med_rep.t())
        
        y_gt_tmp = torch.zeros((batch_num, med_vocab_size)).numpy()

        for index in range(batch_num):
            y_gt_tmp[index, med_index[index]] = 1
            
        for i in range(len(y_gt_tmp)):
            y_gt.append(y_gt_tmp[i])
            
        loss_bce = binary_cross_entropy_with_logits(output, bce_target)
        loss_multi = multilabel_margin_loss(
            torch.sigmoid(output), multi_target)

        output = torch.sigmoid(output).detach().cpu().numpy()
        
        for i in range(len(output)):
            y_pred_prob.append(output[i])

        accum_loss_multi += loss_multi.item()
        accum_loss_bce += loss_bce.item()

        loss_rec = args.bce_weight * loss_bce + \
            (1-args.bce_weight) * loss_multi
        accum_loss_rec += loss_rec.item()

        # Optimize memory usage by avoiding deepcopy
        y_pred_binary = (output >= 0.5).astype(np.float32)

        for i in range(len(y_pred_binary)):
            y_pred.append(y_pred_binary[i])
            y_pred_label_tmp = np.where(y_pred_binary[i] == 1)[0]
            y_pred_label_tmp = sorted(y_pred_label_tmp)
            y_pred_label.append(y_pred_label_tmp)
            smm_record.append([y_pred_label_tmp])
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        y_gt_array = np.array(y_gt)
        y_pred_array = np.array(y_pred)
        y_pred_prob_array = np.array(y_pred_prob)
        
        def jaccard(y_gt, y_pred):
            score = []
            for b in range(y_gt.shape[0]):
                target = np.where(y_gt[b] == 1)[0]
                out_list = np.where(y_pred[b] == 1)[0]
                inter = set(out_list) & set(target)
                union = set(out_list) | set(target)
                jaccard_score = 0 if union == 0 else len(inter) / len(union)
                score.append(jaccard_score)
            return np.mean(score)
        
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            y_gt_array, y_pred_array, y_pred_prob_array
        )
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)

    accum_loss_multi /= len(L)
    accum_loss_bce /= len(L)
    accum_loss_rec /= len(L)

    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_path)

    output_str = '\nDDI Rate: {:.4f}, Jaccard: {:.4f}, PRAUC: {:.4f}, ' +\
        'AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}\n'
    llprint(output_str.format(
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p),
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))

    print("REC Loss: {:.5f}\tBCE loss:{:.5f}\t Multi loss:{:.5f}\tTime: {:.5f}".format(
        accum_loss_rec, accum_loss_bce, accum_loss_multi, time.time() - start_time))

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), \
        np.mean(avg_r), np.mean(avg_f1), med_cnt / \
        visit_cnt, accum_loss_rec, accum_loss_bce, accum_loss_multi


def train(
        epoch,
        dataloader, med_voc, batch_size,  bce_weight, med_vocab_size, input_med_rep,
        mapping_tensor, num_atc3_groups, bce_weight_2):

    criterion = binary_cross_entropy_with_logits

    text2latent.train()
    mol2latent.train()
    aggregator.train()
    global_encoder.train()

    dsp_aggregator.train()
    proj_dsp.train()

    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader

    print("Start training!")
    start_time = time.time()
    accum_loss_rec = 0
    accum_loss_multi, accum_loss_bce = 0, 0

    for step, batch in enumerate(L):
        global_embeddings = global_encoder(**MPNN_drug_data)
        global_embeddings = torch.mm(average_projection, global_embeddings)

        records_dps_txt = batch[0].to(device)
        records_med_txt = batch[1].to(device)
        med_index = batch[2]
        records_dia_txt = batch[3].to(device)
        records_pro_txt = batch[4].to(device)
        records_sym_txt = batch[5].to(device)

        bce_target = batch[6].to(device)
        multi_target = batch[7].to(device)

        his_med_embeddings = batch[8].to(device)
        his_mask = batch[9].to(device)

        med_index = [list(map(int,  elem.split(" "))) for elem in med_index]

        description_repr_input = records_dps_txt

        attn_dsp = dsp_aggregator(description_repr_input, torch.cat([torch.unsqueeze(records_sym_txt, 1), torch.unsqueeze(
            records_dia_txt, 1), torch.unsqueeze(records_pro_txt, 1)], dim=1), mask=None)

        description_repr = torch.cat(
            [description_repr_input, attn_dsp], dim=-1)

        description_repr = proj_dsp(description_repr)

        his_med_repr, _ = aggregator(
            description_repr, his_med_embeddings, mask=his_mask)
        description_repr = text2latent(description_repr)

        # GNN+LLM => encode medications
        med_rep = torch.cat([input_med_rep, global_embeddings], dim=1)

        med_rep = mol2latent(med_rep)

        description_repr = description_repr + his_med_repr

        result = torch.mm(description_repr, med_rep.t())

        def compute_loss(result, bce_target, multi_target, bce_weight_f):
            sigmoid_res = torch.sigmoid(result)
            loss_bce = binary_cross_entropy_with_logits(result, bce_target)
            loss_multi = multilabel_margin_loss(sigmoid_res, multi_target)
            loss_rec = bce_weight_f * loss_bce + (1-bce_weight_f) * loss_multi
            return loss_bce, loss_multi, loss_rec

        loss_bce_atc4, loss_multi_atc4, loss_atc4 = compute_loss(
            result, bce_target, multi_target, bce_weight)

        def optimized_atc_aggregation(result_atc4, bce_target, mapping_tensor, num_atc3_groups):
            device = result_atc4.device
            batch_size = result_atc4.shape[0]

            index = mapping_tensor.expand(batch_size, -1)

            result_atc_agg = torch.zeros(
                (batch_size, num_atc3_groups), device=device)
            result_atc_agg.scatter_add_(dim=1, index=index, src=result_atc4)

            bce_target_atc_agg = torch.zeros(
                (batch_size, num_atc3_groups), device=device)
            bce_target_atc_agg.scatter_add_(dim=1, index=index, src=bce_target)
            bce_target_atc_agg = (bce_target_atc_agg > 0).float()

            # Vectorized multi_target generation
            max_len = num_atc3_groups
            multi_target_atc_agg = torch.full(
                (batch_size, max_len), -1, dtype=torch.long, device=device)

            mask = (bce_target_atc_agg > 0)
            # Create positions for each element: 0, 1, 2... for active elements in each row
            positions = mask.long().cumsum(dim=1) - 1
            
            # Get indices where mask is True
            r_indices, c_indices = mask.nonzero(as_tuple=True)
            
            # Get the target positions in output
            pos_indices = positions[r_indices, c_indices]
            
            # Scatter the column indices (targets) into the output
            multi_target_atc_agg[r_indices, pos_indices] = c_indices

            return result_atc_agg, bce_target_atc_agg, multi_target_atc_agg

        result_atc3, bce_target_atc3, multi_target_atc3 = optimized_atc_aggregation(
            result,
            bce_target,
            mapping_tensor_cached,
            num_atc3_groups_cached
        )

        loss_bce_atc3, loss_multi_atc3, loss_atc3 = compute_loss(
            result_atc3, bce_target_atc3, multi_target_atc3, bce_weight_2)

        accum_loss_rec += loss_atc4.item()
        accum_loss_multi += loss_multi_atc4.item()
        accum_loss_bce += loss_bce_atc4.item()

        def get_curriculum_weights(epoch, total_epochs, method="cosine_decay"):
            progress = (epoch - 1) / (total_epochs - 1)  # 0 to 1
            
            if method == "linear":
                atc3_w = 1 - progress
                atc4_w = progress
                
            elif method == "cosine_decay":
                atc3_w = 0.5 * (1 + math.cos(math.pi * progress))
                atc4_w = 1 - atc3_w
                
            return atc3_w, atc4_w

        if args.method_type == "default":
            loss = loss_atc3 * args.coef + loss_atc4 * (1 - args.coef)
        elif args.method_type == "curriculum":
            atc3_weight, atc4_weight = get_curriculum_weights(epoch, args.epochs, args.curriculum_method)
            loss = loss_atc3 * atc3_weight + loss_atc4 * atc4_weight
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_([
            *text2latent.parameters(),
            *mol2latent.parameters(), 
            *aggregator.parameters(),
            *global_encoder.parameters(),
            *dsp_aggregator.parameters(),
            *proj_dsp.parameters()
        ], max_norm=1.0)
            
        optimizer.step()
        optimizer.zero_grad()

        if step % 1000 == 0:
            print('\rtraining step: {} / {}, REC loss: {:.4f},   loss_bce: {:.4f}, loss_multi: {:.4f} '
                  .format(step, len(L), loss.item(),  loss_bce_atc4.item(), loss_multi_atc4.item()))

    accum_loss_rec /= len(L)
    accum_loss_multi /= len(L)
    accum_loss_bce /= len(L)

    print("REC Loss: {:.5f}\tBCE Loss:{:.5f}\tMulti Loss:{:.5f}\tTime: {:.5f}".format(
        accum_loss_rec, accum_loss_bce, accum_loss_multi, time.time() - start_time))

    return accum_loss_rec, accum_loss_multi, accum_loss_bce

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coef", type=float, default=0.5)
    parser.add_argument("--method_type", type=str, default="curriculum")
    parser.add_argument("--curriculum_method", type=str, default="cosine_decay")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--dataspace_path", type=str,
                        default="../data")
    parser.add_argument("--dataset", type=str, default="MIMIC-III")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_scale", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--decay", type=float, default=1e-5)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument("--output_model_dir", type=str,
                        default='saved')

    # for GNN
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--dropout_ratio", type=float, default=0.2)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument('--graph_pooling', type=str, default='mean')
    parser.add_argument("--gnn_dim", type=int, default=64)
    parser.add_argument("--gnn_dp", type=float, default=0.7)

    parser.add_argument("--K_emb_dim", type=int, default=256)
    parser.add_argument('--data_file_name', type=str,
                        default='../data/mimic-iii/output_atc4/records_text_iii.pkl')

    parser.add_argument("--bce_weight", type=float, default=0.95)
    parser.add_argument("--bce_weight_2", type=float, default=0.95)
    parser.add_argument("--med_vocab_size", type=int, default=217)
    
    # max_visit_num: the # of past records in historical information modeling
    parser.add_argument("--max_visit_num", type=int, default=3)
    parser.add_argument("--pt_mode", type=str, default="bio_pt", choices=[
                        "sci_pt", "clinical_pt", "pub_pt", "bio_pt", "bleu_pt"])

    parser.add_argument("--patient_sample", type=str, default="True")
    parser.add_argument("--Train", action="store_true")

    text_dim = 768
    args = parser.parse_args()
    if args.dataset == "MIMIC-III":
        args.med_vocab_size = 217
    elif args.dataset == "MIMIC-IV":
        args.med_vocab_size = 246

    args.output_model_dir = os.path.join(args.output_model_dir, args.dataset, args.method_type, args.pt_mode, str(
        args.coef), str(args.bce_weight), str(args.bce_weight_2), str(args.curriculum_method))

    print("arguments\t", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if "MIMIC-III" in args.dataset:
        dataset_root = os.path.join(
            args.dataspace_path, "mimic-iii/output_atc4")
    elif "MIMIC-IV" in args.dataset:
        dataset_root = os.path.join(
            args.dataspace_path, "mimic-iv/output_atc4")
    else:
        assert False and "invlaid dataset name"
    kwargs = {}

    if "MIMIC-III" in args.dataset:
        molecule_path = os.path.join(dataset_root, "atc4toSMILES_iii.pkl")
        voc_path = os.path.join(dataset_root, "voc_iii_sym1_mulvisit.pkl")
        ddi_adj_path = os.path.join(dataset_root, "ddi_A_iii.pkl")
        ori_data_path = os.path.join(dataset_root, "records_ori_iii.pkl")
    elif "MIMIC-IV" in args.dataset:
        molecule_path = os.path.join(dataset_root, "atc4toSMILES_iv.pkl")
        voc_path = os.path.join(dataset_root,  "voc_iv_sym1_mulvisit.pkl")
        ddi_adj_path = os.path.join(dataset_root, "ddi_A_iv.pkl")
        ori_data_path = os.path.join(dataset_root, "records_ori_iv.pkl")

    with open(molecule_path, 'rb') as Fin:
        molecule = dill.load(Fin)

    with open(ori_data_path, 'rb') as Fin:
        origin_data = dill.load(Fin)

    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)

    diag_voc, pro_voc, med_voc = \
        voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word)
    )

    average_projection, smiles_list = \
        buildPrjSmiles(molecule, med_voc.idx2word)
    average_projection = average_projection.to(device)
    molecule_graphs = graph_batch_from_smile(smiles_list)
    molecule_forward = {'batched_data': molecule_graphs.to(device)}
    molecule_para = {
        'num_layer': args.num_layer, 'emb_dim': args.gnn_dim, 'graph_pooling': args.graph_pooling,
        'drop_ratio': args.gnn_dp, 'gnn_type': 'gin', 'virtual_node': False
    }

    MPNN_drug_data = molecule_forward

    if args.dataset == "MIMIC-III":
        train_dataset = MIMIC_III_Datasets_text_final(
            root=dataset_root, pt_mode=args.pt_mode,  max_visit_num=args.max_visit_num)
        valid_dataset = MIMIC_III_Datasets_text_Eval_final(
            root=dataset_root, pt_mode=args.pt_mode,  max_visit_num=args.max_visit_num)
        test_dataset = MIMIC_III_Datasets_text_Test_final(
            root=dataset_root, pt_mode=args.pt_mode,  max_visit_num=args.max_visit_num)
    elif args.dataset == "MIMIC-IV":
        train_dataset = MIMIC_IV_Datasets_text_final(
            root=dataset_root, pt_mode=args.pt_mode,  max_visit_num=args.max_visit_num)
        valid_dataset = MIMIC_IV_Datasets_text_Eval_final(
            root=dataset_root, pt_mode=args.pt_mode,  max_visit_num=args.max_visit_num)
        test_dataset = MIMIC_IV_Datasets_text_Test_final(
            root=dataset_root, pt_mode=args.pt_mode,  max_visit_num=args.max_visit_num)
    dataloader_class = torch_DataLoader

    train_loader = dataloader_class(train_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = dataloader_class(valid_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    test_loader = dataloader_class(test_dataset, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    global_encoder = GNNGraph(**molecule_para).to(device)

    dsp_aggregator = AdjAttenAgger_no_projection(
        text_dim, text_dim, args.K_emb_dim, device
    ).to(device)

    proj_dsp = nn.Sequential(
        nn.Linear(2*text_dim, text_dim),
        nn.GELU(),
        nn.Dropout(args.dropout_ratio),
    ).to(device)

    aggregator = AdjAttenAgger(
        text_dim, text_dim, args.K_emb_dim, device
    ).to(device)

    text2latent = MLP(text_dim, [args.K_emb_dim, args.K_emb_dim], batch_norm=False,
                      activation="gelu", dropout=args.dropout_ratio).to(device)
    mol2latent = MLP(text_dim+64, [args.K_emb_dim, args.K_emb_dim],
                     batch_norm=False, activation="gelu", dropout=args.dropout_ratio).to(device)

    model_param_group = [
        {"params": text2latent.parameters(), "lr": args.lr * args.lr_scale},
        {"params": mol2latent.parameters(), "lr": args.lr * args.lr_scale},
        {"params": aggregator.parameters(), "lr": args.lr * args.lr_scale},
        {"params": global_encoder.parameters(), "lr": args.lr * args.lr_scale},
        {"params": dsp_aggregator.parameters(), "lr": args.lr * args.lr_scale},
        {"params": proj_dsp.parameters(), "lr": args.lr * args.lr_scale},
    ]

    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    print("parameters", get_n_params(text2latent)+get_n_params(mol2latent)+get_n_params(aggregator)+get_n_params(global_encoder) +
          get_n_params(dsp_aggregator)+get_n_params(proj_dsp))

    optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
    optimal_loss = 1e10

    best_epoch = 0
    best_ja = 0

    kwargs['batch_size'] = args.batch_size
    kwargs['bce_weight'] = args.bce_weight
    kwargs['med_vocab_size'] = args.med_vocab_size
    kwargs['bce_weight_2'] = args.bce_weight_2
    save_file = os.path.join(
        dataset_root, args.pt_mode, "med_output_tensor.pt")

    input_med_rep = torch.load(save_file)
    input_med_rep = input_med_rep.detach().to(device)

    kwargs['input_med_rep'] = input_med_rep

    print("Pre-computing ATC mapping...")

    def create_atc_mapping_cached(med_voc, device):
        med_vocab_size = len(med_voc.idx2word)
        atc4_map = {}
        mapping_list = [0] * med_vocab_size

        for i in range(med_vocab_size):
            atc3_code = med_voc.idx2word[i][:4]
            if atc3_code not in atc4_map:
                atc4_map[atc3_code] = len(atc4_map)
            mapping_list[i] = atc4_map[atc3_code]

        return torch.tensor(mapping_list, dtype=torch.long, device=device), len(atc4_map)

    mapping_tensor_cached, num_atc3_groups_cached = create_atc_mapping_cached(
        med_voc, device)

    if args.Train:
        best_ja = 0
        for e in range(1, args.epochs+1):
            print("---------Epoch {}-----------".format(e))

            accum_loss_rec, accum_loss_multi, accum_loss_bce = train(
                e, train_loader, med_voc, args.batch_size,  args.bce_weight, args.med_vocab_size, input_med_rep,
                mapping_tensor_cached, num_atc3_groups_cached, args.bce_weight_2)

            model_name = 'Epoch_{}'.format(e)
            print("Valid perfomance")
            ddi_rate_eval, ja_eval, prauc_eval, avg_p_eval, avg_r_eval, avg_f1_eval, avg_med_eval, accum_loss_rec, accum_loss_bce, accum_loss_multi = eval_epoch(
                val_loader, args.batch_size, args.med_vocab_size, input_med_rep)

            if best_ja < ja_eval:
                best_epoch = e
                best_ja = ja_eval
                save_model(save_best=True, epoch=e, model_name=model_name)
            print("best_ja", best_ja)
            print("best_epoch", best_epoch)

        save_best_epoch = best_epoch

        text2latent.load_state_dict(torch.load(open(os.path.join(args.output_model_dir, str(
            save_best_epoch), "text2latent_model_Epoch_{}.pth".format(str(save_best_epoch))), 'rb'), map_location=torch.device('cpu')))
        print("loading text2latent")
        mol2latent.load_state_dict(torch.load(open(os.path.join(args.output_model_dir, str(
            save_best_epoch), "mol2latent_model_Epoch_{}.pth".format(str(save_best_epoch))), 'rb'), map_location=torch.device('cpu')))
        print("loading mol2latent")
        aggregator.load_state_dict(torch.load(open(os.path.join(args.output_model_dir, str(
            save_best_epoch), "aggregator_model_Epoch_{}.pth".format(str(save_best_epoch))), 'rb'), map_location=torch.device('cpu')))
        print("loading aggregator")
        dsp_aggregator.load_state_dict(torch.load(open(os.path.join(args.output_model_dir, str(
            save_best_epoch), "dsp_aggregator_model_Epoch_{}.pth".format(str(save_best_epoch))), 'rb'), map_location=torch.device('cpu')))
        print("loading dsp_aggregator")
        proj_dsp.load_state_dict(torch.load(open(os.path.join(args.output_model_dir, str(
            save_best_epoch), "proj_dsp_model_Epoch_{}.pth".format(str(save_best_epoch))), 'rb'), map_location=torch.device('cpu')))
        print("loading proj_dsp_aggregator")
        global_encoder.load_state_dict(torch.load(open(os.path.join(args.output_model_dir, str(
            save_best_epoch), "global_encoder_model_Epoch_{}.pth".format(str(save_best_epoch))), 'rb'), map_location=torch.device('cpu')))
        print("loading MPNN")
        
        result = []

        if args.patient_sample == "True" and (args.dataset == "MIMIC-III" or "MIMIC-IV"):
            patient_list = []
            initial_patient_index = 0
            split_point = int(len(origin_data) * 2 / 3)
            eval_len = int(len(origin_data[split_point:]) / 2)
            origin_data_test = origin_data[split_point:split_point + eval_len]
            for patient in origin_data_test:
                every_patient = []
                for visit in patient:
                    every_patient.append(initial_patient_index)
                    initial_patient_index = initial_patient_index + 1
                patient_list.append(every_patient)

        for _ in range(10):
            if args.patient_sample == "True" and args.dataset == "MIMIC-III":
                sample_size = np.random.choice(a=len(origin_data_test), size=round(
                    len(origin_data_test) * 0.8), replace=True)
                test_sample = list(sample_size)
                random_list = []
                for sample_idx in test_sample:
                    random_list = random_list + patient_list[sample_idx]

                test_dataset_my = MIMIC_III_Datasets_text_Test_final_my(
                    index_list=random_list, root=dataset_root, pt_mode=args.pt_mode,  max_visit_num=args.max_visit_num)
                test_loader_my = dataloader_class(
                    test_dataset_my, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            elif args.patient_sample == "True" and args.dataset == "MIMIC-IV":
                sample_size = np.random.choice(a=len(origin_data_test), size=round(
                    len(origin_data_test) * 0.8), replace=True)
                test_sample = list(sample_size)
                random_list = []
                for sample_idx in test_sample:
                    random_list = random_list + patient_list[sample_idx]
                test_dataset_my = MIMIC_IV_Datasets_text_Test_final_my(
                    index_list=random_list, root=dataset_root, pt_mode=args.pt_mode,  max_visit_num=args.max_visit_num)
                test_loader_my = dataloader_class(
                    test_dataset_my, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            else:
                test_sampler_my = torch.utils.data.RandomSampler(
                    data_source=test_dataset, replacement=True, num_samples=round(len(test_dataset) * 0.8))
                test_loader_my = dataloader_class(
                    dataset=test_dataset, sampler=test_sampler_my, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            with torch.set_grad_enabled(False):
                ddi_rate, ja, prauc, avg_p_eval, avg_r_eval, avg_f1, avg_med, accum_loss_rec, accum_loss_bce, accum_loss_multi = eval_epoch(
                    test_loader_my, args.batch_size, args.med_vocab_size, input_med_rep)
                result.append([ddi_rate, ja, avg_f1, prauc, avg_med])

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print(outstring)

        # save the outstring to a file
        with open("outstring.txt", "a") as f:
            f.write(f"dataset: {args.dataset} pt_mode: {args.pt_mode} lr: {args.lr} batch_size: {args.batch_size} bce_weight: {args.bce_weight} bce_weight_2: {args.bce_weight_2}\n")
            f.write(outstring)
            f.write("\n\n")
