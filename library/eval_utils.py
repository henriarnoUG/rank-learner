import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from metrics import *
from models import *


####################################################################################

class EvalDataset(Dataset):
    def __init__(self, df, confounders):
        self.df = df
        self.confounders = confounders

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.confounders].astype(np.float32).to_numpy(), dtype=torch.float32) 
        cate = torch.tensor(float(row['cate']), dtype=torch.float32)
        M0 = torch.tensor(float(row['M0']), dtype=torch.float32)
        M1 = torch.tensor(float(row['M1']), dtype=torch.float32)
        
        return x, cate, M0, M1

####################################################################################

def get_estimates_cate(cate_model, m0_model, m1_model, test_loader, device):
    cate_model.eval(); m0_model.eval(); m1_model.eval()
    
    # collectors
    dr_hats, t_hats = [], []
    cate_true, M0_true, M1_true = [], [], []
    
    # loop over test loader
    with torch.inference_mode():
        for x, cate, M0, M1 in test_loader:

            # forward pass
            x = x.to(device)
            dr_hat    = cate_model(x).squeeze(-1)  
            m0_hat    = m0_model(x).squeeze(-1)    
            m1_hat    = m1_model(x).squeeze(-1)    

            # collect
            t_hat = m1_hat - m0_hat
            dr_hats.append(dr_hat)
            t_hats.append(t_hat)
            cate_true.append(cate.squeeze(-1))
            M0_true.append(M0.squeeze(-1))
            M1_true.append(M1.squeeze(-1))
    
    # store in df dataframe
    to_np = lambda parts: torch.cat(parts, dim=0).detach().cpu().numpy()
    df_eval = pd.DataFrame({
        "DR_hat": to_np(dr_hats), 
        "T_hat":   to_np(t_hats),       
        "cate":    to_np(cate_true),   
        "M0":      to_np(M0_true),
        "M1":      to_np(M1_true)})

    return df_eval


def get_estimates_rankers(plugin_model, rank_learner, test_loader, device):
    plugin_model.eval(); rank_learner.eval()
    
    # collectors
    plugin_hats, ranker_hats = [], []
    cate_true, M0_true, M1_true = [], [], []
    
    # loop over test loader
    with torch.inference_mode():
        for x, cate, M0, M1 in test_loader:
            
            # forward pass
            x = x.to(device)
            plugin_hat  = plugin_model(x).squeeze(-1)      
            ranker_hat  = rank_learner(x).squeeze(-1)    

            # collect
            plugin_hats.append(plugin_hat)
            ranker_hats.append(ranker_hat)
            
            cate_true.append(cate.squeeze(-1))
            M0_true.append(M0.squeeze(-1))
            M1_true.append(M1.squeeze(-1))
    
    # dataframe
    to_np = lambda parts: torch.cat(parts, dim=0).detach().cpu().numpy()
    df_eval = pd.DataFrame({
        "plugin_score":       to_np(plugin_hats), 
        "rank_learner_score": to_np(ranker_hats),       
        "cate":    to_np(cate_true),   
        "M0":      to_np(M0_true),
        "M1":      to_np(M1_true)})

    return df_eval
    
####################################################################################

def compute_metrics_cate(eval_df):
    # collector
    results = []
    
    # loop over scores
    for score in ['cate','DR_hat', 'T_hat']:
        ranked = eval_df.sort_values(score, ascending=False).copy()
    
        # eval
        row = {"score": score}
        row["autoc"]   = autoc(ranked)
        row["mean_pv"] = mean_policy_value(ranked)
        ranked = ranked.assign(est=ranked[score])
        row['pehe'] = pehe(ranked)
        results.append(row)
    
    # store
    results = pd.DataFrame(results)
    return results



def compute_metrics_rankers(eval_df):
    # collector
    results = []
    
    # loop over scores
    for score in ['plugin_score', 'rank_learner_score']:
        ranked = eval_df.sort_values(score, ascending=False).copy()
    
        # eval
        row = {"score": score}
        row["autoc"]   = autoc(ranked)
        row["mean_pv"] = mean_policy_value(ranked)
        results.append(row)
    
    # store
    results = pd.DataFrame(results)
    return results