import copy
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# from sklearn.cluster import KMeans

# from trainer.config import cfg
import numpy as np
import pandas as pd
from scipy import sparse as sp
import anndata as ad
from torch.utils.data import Dataset
import scanpy as sc
import torch
#

# class MyDataset(Dataset):
#     """Operations with the datasets."""

#     def __init__(self, ad, use_label=True):
#         """
#         Args:
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.data = np.array(ad.X)  # anndata格式存表达值和细胞类型
#         self.data = torch.tensor(self.data, dtype=torch.float)
#         if use_label:
#             self.data_cls = pd.Categorical(ad.obs['cell_type']).codes  #
#         else:
#             sc.pp.pca(ad)
#             sc.pp.neighbors(ad)
#             X_pca = ad.obsm['X_pca']

#             kmeans = KMeans(n_clusters=cfg.N_CLUSTER, random_state=0)
#             clusters = kmeans.fit_predict(X_pca)
#             ad.obs['kmeans'] = clusters.astype(str)

#             self.data_cls = pd.Categorical(ad.obs['kmeans']).codes
#         self.data_cls = torch.tensor(self.data_cls, dtype=torch.long)

#     def __len__(self):
#         return len(self.data_cls)

#     def __getitem__(self, idx):
#         data = self.data[idx, :]
#         label = self.data_cls[idx]
#         return data, label

def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=False, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        # sc.pp.normalize_total(adata, target_sum=1e4)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)

    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    if adata.n_vars > 5000:
        sc.pp.highly_variable_genes(adata, n_top_genes=5000)
        adata = adata[:, adata.var['highly_variable'] == True].copy()

    return adata


def normalize_and_log(X, X_imp, do_normalize=True, do_log=True, target_sum=1e4, copy=True):
    """
    Chuẩn hóa X bằng scanpy và áp dụng CHÍNH XÁC cùng phép biến đổi lên X_imp.
    - target_sum: giống sc.pp.normalize_total (mặc định 1e4).
    - Yêu cầu X và X_imp có cùng số hàng (cell).
    """

    # Tổng theo cell trên X (trước chuẩn hóa)
    if sp.issparse(X):
        total_counts_X = np.asarray(X.sum(axis=1)).ravel()
    else:
        total_counts_X = X.sum(axis=1).ravel()

    # Tránh chia 0
    safe_total = total_counts_X.copy()
    safe_total[safe_total == 0] = 1.0


    adata = ad.AnnData(X.copy() if copy else X)
    if do_normalize:
        sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed=False)
    if do_log:
        sc.pp.log1p(adata)
    X_norm = adata.X

    #    factor_i = total_counts_X[i] / target_sum  -> X_imp / factor_i
    if do_normalize:
        factors = safe_total / float(target_sum)
        if sp.issparse(X_imp):
            X_imp_scaled = X_imp.multiply(1.0 / factors[:, None])
        else:
            X_imp_scaled = X_imp / factors[:, None]
    else:
        X_imp_scaled = X_imp.copy()

    # 3) Log1p nếu cần
    if do_log:
        if sp.issparse(X_imp_scaled):
            coo = X_imp_scaled.tocoo()
            coo.data = np.log1p(coo.data)
            X_imp_scaled = coo.tocsr()
        else:
            X_imp_scaled = np.log1p(X_imp_scaled)

    X_norm = X_norm.astype(np.float32) if sp.issparse(X_norm) else np.asarray(X_norm, dtype=np.float32)
    X_imp_scaled = X_imp_scaled.astype(np.float32) if sp.issparse(X_imp_scaled) else np.asarray(X_imp_scaled, dtype=np.float32)

    return X_norm, X_imp_scaled


# if __name__ == '__main__':
#     drop_expression_path = r'F:/pythonProject/scGACL/dataset/sim.groups3_counts.csv'
#     celltype_path = r'F:/pythonProject/scGACL/dataset/sim.groups3_groups.csv'
#     drop_rna_ad = sc.read(drop_expression_path).T
#     drop_rna_ad = normalize(drop_rna_ad)
#     print(drop_rna_ad)
#     print(drop_rna_ad.to_df())
#     celltype_df = pd.read_csv(celltype_path, index_col=0)
#     print(celltype_df)
#     new_index = []
#     for index in celltype_df.index:
#         index = str(index)
#         index = 'Cell' + index
#         new_index.append(index)
#     celltype_df.index = new_index
#     cells = drop_rna_ad.obs_names.tolist()
#     celltype_df = celltype_df.loc[cells, :]
#     drop_rna_ad.obs['cell_type'] = celltype_df.loc[:, 'cell_type']

#     print(drop_rna_ad)
#     print(drop_rna_ad.obs)
#     sc.write(r'F:/pythonProject/scGACL/dataset/sim.groups1_preprocess.h5ad', drop_rna_ad)
# if __name__ == '__main__':
#     drop_expression_path = r'F:\simulate_dataset\dataset1\counts.csv'
#     celltype_path = r'F:\simulate_dataset\dataset1\groups.csv'
#     drop_rna_ad = sc.read(drop_expression_path).T
#     drop_rna_ad = normalize(drop_rna_ad)
#     print(drop_rna_ad)
#     print(drop_rna_ad.to_df())
#     celltype_df = pd.read_csv(celltype_path, index_col=0)
#     print(celltype_df)
#
#     cells = drop_rna_ad.obs_names.tolist()
#     celltype_df = celltype_df.loc[cells, :]
#     drop_rna_ad.obs['cell_type'] = celltype_df.loc[:, 'Group']
#
#     print(drop_rna_ad)
#     print(drop_rna_ad.obs)


if __name__ == '__main__':
    # drop_rate = 20
    for drop_rate in range(20, 95, 10):
        print(f'---Drop rate {drop_rate}----------')
        drop_expression_path = f'./sim.Tung/sim.Tung.drop{drop_rate}/SplatDrop_counts.csv'
        true_express_path = f'./sim.Tung/sim.Tung.drop{drop_rate}/SplatDrop_trueCounts.csv'
        
        print(f'Loading {drop_expression_path}....')
        drop_rna = pd.read_csv(drop_expression_path, index_col=0, header=0)
        print(f'Loading success....')
        print(f'Loading {true_express_path}....')
        true_rna = pd.read_csv(true_express_path, index_col=0, header=0)
        print(f'Loading success....')

        if not drop_rna.index.equals(true_rna.index) or not drop_rna.columns.equals(true_rna.columns):
            raise ValueError('Dropout and true count matrices must have the same genes and cells.')
        
        genes = drop_rna.index.astype(str)
        cells = drop_rna.columns.astype(str)

        x = drop_rna.T.values
        x_true = true_rna.T.values
        
        x, x_true = normalize_and_log(x, x_true)
        
        # drop_rna_ad = ad.AnnData(
        #     x,
        #     obs=pd.DataFrame(index=cells),
        #     var=pd.DataFrame(index=genes),
        # )
        # drop_rna_ad.obs_names_make_unique()
        # drop_rna_ad.var_names_make_unique()
        
        # print(drop_rna_ad)
        # print(f'First cells: {drop_rna_ad.obs_names[:5].tolist()}')
        # print(f'First genes: {drop_rna_ad.var_names[:5].tolist()}')
        pd.DataFrame(x, index=cells, columns=genes).to_csv(
                    f'./sim.Tung/sim.Tung.drop{drop_rate}/SplatDrop_norm.csv'
                )
        pd.DataFrame(x_true, index=cells, columns=genes).to_csv(
            f'./sim.Tung/sim.Tung.drop{drop_rate}/SplatDrop_trueCounts_norm.csv'
        )
        # sc.write(f'./sim.Tung/sim.Tung.drop{drop_rate}/preprocess_norm.h5ad', drop_rna_ad)
        print('-'*20)
        if drop_rate == 20:
            break
    
