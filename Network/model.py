import os, random, pickle
import numpy as np
import torch
import scipy.sparse as ssp
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import List, Union
from tqdm import tqdm
from sklearn.decomposition import PCA
import torch.nn.functional as F

"""
Dataset & models for InterLabelGO+  – cleaned version.
Only the *mmstie* dual‑channel embeddings are supported in this edition.
"""

# NOTE: This file was cleaned on 2025‑05‑16 to remove hard‑coded paths and debug prints.


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Default ESM representation layers (kept in one place)
DEFAULT_REPR_LAYERS = [34, 35, 36]   # ESM default layers (kept in one place)

# A small helper that maps each embedding type to its default sub‑folder.
EMB_SUBDIR = {
    "esm":      "/scratch0/cafa5_small/esm/",
    "esm_mean": "/scratch0/cafa5_small/esm_mean/",
    "text":      "/scratch0/cafa5_small/prot2text/text_embeddings",

    "t5":       "embeddings/t5",
    "mmsite":   "embeddings/mmstie",

}




class InterlabelGODataset(Dataset):
    def __init__(self,
        features_dir: str,
        embedding_type: str,                # "esm" | "t5" | "mmstie"
        names_npy: str,
        labels_npy: str | None = None,
        repr_layers: list | None = None,
        low_memory: bool = False,
        pca: bool = True,
        do_normalize: bool = True,
    ):

        # print("features_dir", features_dir)
        # exit()
        self.features_dir = features_dir
        self.names_npy = names_npy
        self.repr_layers = repr_layers if repr_layers is not None else DEFAULT_REPR_LAYERS

        embedding_type = embedding_type.lower()
        assert embedding_type in {"esm", "esm_mean", "t5", "mmsite", "text"}, \
            f"Unsupported embedding_type '{embedding_type}'"
        self.low_memory = low_memory
        self.feature_cache = dict()
        self.temp = dict()
        self.embedding_type = embedding_type
        self.pca = pca


        self.do_normalize = do_normalize
        self.mean_ = None
        self.std_ = None

        if labels_npy is None:
            self.prediction = True
        else:
            self.prediction = False
            self.labels_npy = labels_npy

        # load names, labels
        self.names = np.load(self.names_npy)

        if not self.prediction:
            self.labels = self.load_labels(self.labels_npy)

        # --------------------------------------------------------------
        # Pre-load embeddings according to embedding_type
        # --------------------------------------------------------------

        if not self.low_memory:
            if self.embedding_type == "mmsite":
                for name in tqdm(self.names, desc="Loading mmsite embeddings"):
                    self.feature_cache[name], self.temp[name] = self.load_featureMM(name)
            elif self.embedding_type == "t5":
                for name in tqdm(self.names, desc="Loading T5 embeddings"):
                    self.feature_cache[name] = self.load_featureT5(name)
            elif self.embedding_type in {"esm", "esm_mean"}:
                subdir = EMB_SUBDIR[self.embedding_type]
                for name in tqdm(self.names, desc=f"Loading {self.embedding_type} embeddings"):
                    self.feature_cache[name] = self._load_feature_file(
                        os.path.join(self.features_dir, subdir), name
                    )



    def load_labels(self, labels_npy:str)->np.ndarray:
        """
        Load labels from npy or npz file.

        Args:
            labels_npy (str): path to npy or npz file

        Raises:
            Exception: Unknown label file format

        Returns:
            np.ndarray: labels
        """
        if labels_npy.endswith(".npy"):
            labels = np.load(labels_npy)
        elif labels_npy.endswith(".npz"):
            labels = ssp.load_npz(labels_npy).toarray()
        else:
            raise Exception("Unknown label file format")
        labels = torch.from_numpy(labels).float()
        return labels
    

    def __len__(self):
        # print(len(self.names))
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]

        # lazily load if needed
        if name not in self.feature_cache:
            if self.embedding_type == "mmsite":
                self.feature_cache[name], self.temp[name] = self.load_featureMM(name)
            elif self.embedding_type == "t5":
                self.feature_cache[name] = self.load_featureT5(name)
            else:
                subdir = EMB_SUBDIR[self.embedding_type]
                self.feature_cache[name] = self._load_feature_file(
                    os.path.join(self.features_dir, subdir), name
                )

        if self.prediction:
            if self.embedding_type == "mmsite":
                return name, self.feature_cache[name], self.temp[name]
            else:
                return name, self.feature_cache[name]
        else:   
            label = self.labels[idx]
            if self.embedding_type == "mmsite":

                return name, self.feature_cache[name], self.temp[name], label
            else:
                return name, self.feature_cache[name], label
        
    # def parse_uniprot_id(self, pdb_filename):
    #     """
    #     Extracts the UniProt ID from a filename of the form:
    #     AF-<UniProtID>-F1-model_v4.pdb
    #     E.g. "AF-V9HWF5-F1-model_v4.pdb" -> "V9HWF5"
    #     """
    #     base, _ = os.path.splitext(pdb_filename)            # e.g. "AF-V9HWF5-F1-model_v4"
    #     parts = base.split("-")                             # ["AF", "V9HWF5", "F1", "model_v4"]
    #     uniprot_id = parts[1]                               # "V9HWF5"
    #     return uniprot_id


    def load_featureMM(self, name:str)->np.ndarray:
        seq_mat  = np.load(os.path.join(
            EMB_SUBDIR['esm'], f'{name}.npy'),
            allow_pickle=True).item()["embedding"]
        text_mat = np.load(os.path.join(
             EMB_SUBDIR['text'], f'{name}.npy'),
            allow_pickle=True).item()["embedding"]
            

        # not (L×D)

        seq_vec = seq_mat.astype(np.float32)



        text_vec = text_mat.astype(np.float32)

        return torch.from_numpy(seq_vec).float(), torch.from_numpy(text_vec).float()

    


    def load_featureT5(self, name:str)->np.ndarray:
        """
        Load feature from npy file.

        Args:
            name (str): name of the feature

        Returns:
            np.ndarray: feature
        """
        subdir = EMB_SUBDIR["T5"]
        features = np.load(os.path.join(self.features_dir, subdir, f"{name}.npy"), allow_pickle=True)
        if isinstance(features, np.ndarray) and features.ndim == 2:
            features = features.mean(axis=0).astype(np.float32)
        return torch.from_numpy(features).float()


    # ------------------------------------------------------------------
    # Helper: pool per‑token matrices (L × D) into a single vector (D)
    # ------------------------------------------------------------------
    def _pool_token_matrix(self, mat: np.ndarray) -> np.ndarray:
        """Default pooling = mean over sequence length dimension."""
        return mat.mean(axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Helper: *average* the selected layer means into a single vector
    # ------------------------------------------------------------------
    def _avg_layer_means(self, esm_dict: dict) -> np.ndarray:
        """
        Average the 'mean' vectors for self.repr_layers instead of concatenating.
        Produces a single 1280‑D vector just like mean‑pooling over tokens.
        """
        return np.stack([esm_dict['mean'][l] for l in self.repr_layers]).mean(axis=0).astype(np.float32)

    def load_feature(self, name:str)->np.ndarray:
        """
        Load feature from npy file.

        Args:
            name (str): name of the feature

        Returns:
            np.ndarray: feature
        """
        subdir = EMB_SUBDIR.get("ESM", "")
        features = np.load(os.path.join(self.features_dir, subdir, f"{name}.npy"), allow_pickle=True).item()
        final_features = self._avg_layer_means(features)
        return torch.from_numpy(final_features).float()

    def _load_feature_file(self, directory: str, name: str) -> torch.Tensor:
        """Load a single .npy and return a float32 tensor."""
        path = os.path.join(directory, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        arr = np.load(path, allow_pickle=True)

        # Fast path: esm_mean already stored as a single vector
        if self.embedding_type == "esm_mean":
            if isinstance(arr, np.ndarray):
                if arr.dtype == object:
                    arr = arr.item()["embedding"]
                # if stored as dict with key 'embedding'
                elif arr.ndim == 0 and isinstance(arr.item(), dict):
                    arr = arr.item()["embedding"]
                return torch.from_numpy(arr.astype(np.float32)).float()

        # Fast path: text embeddings are stored as {"name": id, "embedding": vector}
        if self.embedding_type == "text":
            if isinstance(arr, np.ndarray):
                if arr.dtype == object:
                    arr = arr.item()["embedding"]
                elif arr.ndim == 0 and isinstance(arr.item(), dict):
                    arr = arr.item()["embedding"]
                if arr.ndim == 2:                    # (L, 768) → mean-pool
                    arr = arr.mean(axis=0)
                return torch.from_numpy(arr.astype(np.float32)).float()

        # ── Only ESM uses the flexible loader below ─────────────────────
        if self.embedding_type == "esm":
            # object array -> dict
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                arr = arr.item()

            # Dict formats ------------------------------------------------
            if isinstance(arr, dict):
                if "mean" in arr:                 # layer‑means dict
                    arr = self._avg_layer_means(arr)
                elif "embedding" in arr:          # (L, D) matrix
                    arr = self._pool_token_matrix(arr["embedding"])
                else:
                    raise ValueError(f"Unknown keys in ESM file {path}: {arr.keys()}")

            # Plain ndarray formats --------------------------------------
            elif isinstance(arr, np.ndarray):
                if arr.ndim == 2:                 # (L, D) -> pool
                    arr = self._pool_token_matrix(arr)
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
            else:
                raise TypeError(f"Unsupported ESM data type in {path}: {type(arr)}")

        else:
            # Original behaviour for T5 and mmstie
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                arr = self._avg_layer_means(arr.item())
            elif isinstance(arr, np.ndarray):
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
            else:
                raise TypeError(f"Unsupported data type in {path}: {type(arr)}")

        return torch.from_numpy(arr).float()


# 假设 feature_cache 和 temp 已经提供
# feature_cache = {name: embedding_1024[i] for i, name in enumerate(names)}
# temp = {name: reduced_2048[i] for i, name in enumerate(names)}

# class AttentionFusion(nn.Module):
#     def __init__(self, embedding_dim):
#         super(AttentionFusion, self).__init__()
#         # 注意力权重计算层
#         self.attention_fc = nn.Linear(embedding_dim * 2, 2)  # 两个嵌入维度拼接后映射到两个权重

#     def forward(self, feature1, feature2):
#         # 将两个嵌入向量拼接
#         concatenated = torch.cat((feature1, feature2), dim=-1)  # [1024 + 1024]
#         # 计算注意力权重
#         attention_weights = self.attention_fc(concatenated)  # [batch_size, 2]
#         attention_weights = F.softmax(attention_weights, dim=-1)  # 归一化
#         # 加权融合
#         fused_feature = (
#             attention_weights[:, 0:1] * feature1 + attention_weights[:, 1:2] * feature2
#         )
#         return fused_feature

# class InterlabelGODatasetWindow(Dataset):
#     def __init__(self,
#         features_dir:str,
#         fasta_dict:dict,
#         repr_layers:list=[34, 35, 36],
#         window_size:int=50,
#     ):
#         self.features_dir = features_dir
#         self.fasta_dict = fasta_dict
#         self.repr_layers = repr_layers
#         self.window_size = window_size
#         self.data_list = self.load_data()

#     def load_data(self):
#         data_dict = {}
#         for name, seq in self.fasta_dict.items():
#             # truncate the sequence to the first 1000 amino acids, because the esm model only accept 1000 amino acids
#             if len(seq) > 1000:
#                 seq = seq[:1000]
#             features = np.load(os.path.join(self.features_dir, name + '.npy'),allow_pickle=True).item()
#             seq_len = len(seq)
#             # create windows
#             for i in range(0, seq_len-self.window_size+1):
#                 start = i
#                 end = i + self.window_size
#                 final_features = []
#                 for layer in self.repr_layers:
#                     final_features.append(features['per_tok'][layer][start:end].mean(axis=0))
#                 final_features = np.concatenate(final_features)
#                 final_features = torch.from_numpy(final_features).float()
#                 data_dict[name + '_' + str(start) + '-' + str(end)] = final_features
#         return list(data_dict.items())

#     def __len__(self):
#         return len(self.data_list)
    
#     def __getitem__(self, idx):
#         name, feature = self.data_list[idx]
#         return name, feature

# class Residual(nn.Module):

#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x):
#         return x + self.fn(x)


# class InterLabelResNet(nn.Module):
#     def __init__(self, 
#         aspect:str=None, # aspect of the GO terms
#         layer_list:list=[1024], # layers of dnn network, example [512, 256, 128]
#         embed_dim:int=2560, # dim of the embedding protein language model
#         go_term_list:List[str]=[], # list of GO terms for prediction
#         dropout:float=0.3, # dropout rate
#         activation:str='elu', # activation function
#         seed:int=42, # random seed
#         prediction_mode:bool=False, # if True, the model will output the prediction of the GO terms
#         add_res:bool=False,
#         ):
#         super(InterLabelResNet, self).__init__()
#         self.aspect = aspect
#         self.layer_list = layer_list
#         self.embed_dim = embed_dim
#         self.go_term_list = go_term_list
#         self.vec2go_dict = self.get_vec2go()
#         self.class_num = len(go_term_list)
#         self.dropout = dropout
#         self.activation = activation
#         self.seed = seed
#         self.prediction_mode = prediction_mode
#         self.add_res = add_res

#         # bach normalization for the input
#         self.bn1 = nn.BatchNorm1d(embed_dim)
#         self.bn2 = nn.BatchNorm1d(embed_dim)
#         self.bn3 = nn.BatchNorm1d(embed_dim)

#         # Define DNN branches
#         self.branch1 = self._build_dnn_branch(embed_dim)
#         self.branch2 = self._build_dnn_branch(embed_dim)
#         self.branch3 = self._build_dnn_branch(embed_dim)
        
#         # concat dense layer
#         self.concat_layer = nn.Sequential(
#             nn.Linear(layer_list[-1]*3, (layer_list[-1])),
#             self.get_activation(activation),
#             nn.Dropout(dropout),
#             nn.BatchNorm1d((layer_list[-1])),
#         )


#         if self.add_res:
#             self.res = Residual(
#                 nn.Sequential(
#                     nn.Linear(layer_list[-1], layer_list[-1]),
#                     self.get_activation(activation),
#                     nn.Dropout(0.1),
#                     nn.BatchNorm1d((layer_list[-1])),
#                 )
#             )

#         self.output_layer = nn.Sequential(
#             nn.Linear((layer_list[-1]), self.class_num),
#             #nn.Sigmoid(),
#         )

#         # initialize weights
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)


#     def _build_dnn_branch(self, embed_dim):
#         layers = []
#         for i, layer in enumerate(self.layer_list):
#             layers.append(nn.Linear(embed_dim if i == 0 else self.layer_list[i - 1], layer))
#             layers.append(self.get_activation(self.activation))
#             layers.append(nn.Dropout(self.dropout))
#             layers.append(nn.BatchNorm1d(layer))
#         return nn.Sequential(*layers) # * is used to unpack the list for the nn.Sequential

#     def forward(self, inputs):
#         # print(self.embed_dim)
#         # print(inputs.shape)
#         x1 = inputs[:, :self.embed_dim]
#         x2 = inputs[:, self.embed_dim:2*self.embed_dim]
#         x3 = inputs[:, 2*self.embed_dim:]


#         # batch normalization for each branch
#         x1 = self.bn1(x1)
#         x2 = self.bn2(x2)
#         x3 = self.bn3(x3)
#         # print(inputs.shape)
#         # print(x1.shape, x2.shape, x3.shape)
#         # 
        
#         x1 = self.branch1(x1)

#         x2 = self.branch2(x2)
#         x3 = self.branch3(x3)




#         x = torch.cat((x1, x2, x3), dim=1)
#         x = self.concat_layer(x)

#         if self.add_res:
#             x = self.res(x)

#         y_pred = self.output_layer(x)

#         if self.prediction_mode:
#             y_pred = torch.sigmoid(y_pred)

#         return y_pred
    
#     def get_vec2go(self):
#         vec2go_dict = dict()
#         for i, go_term in enumerate(self.go_term_list):
#             vec2go_dict[i] = go_term
#         return vec2go_dict
    
#     def get_activation(self, activation:str):
#         activation = activation.lower()
#         if activation == 'relu':
#             return nn.ReLU()
#         elif activation == 'leakyrelu':
#             return nn.LeakyReLU()
#         elif activation == 'tanh':
#             return nn.Tanh()
#         elif activation == 'sigmoid':
#             return nn.Sigmoid()
#         elif activation == 'elu':
#             return nn.ELU()
#         elif activation == 'gelu':
#             return nn.GELU()
#         else:
#             raise ValueError('activation function not supported')
    
#     def save_config(self, save_path:str):
#         config = {
#             'aspect': self.aspect, 
#             'layer_list': self.layer_list, 
#             'embed_dim': self.embed_dim, 
#             'go_term_list': self.go_term_list, 
#             'dropout': self.dropout, 
#             'activation': self.activation,
#             'seed': self.seed, 
#             'add_res': self.add_res,
#             'state_dict': self.state_dict(),
#         }
#         torch.save(config, save_path)
        

#     @staticmethod
#     def load_config(save_path:str):
#         config = torch.load(save_path, map_location=torch.device('cpu'))
#         model = InterLabelResNet(
#             aspect=config['aspect'], 
#             layer_list=config['layer_list'], 
#             embed_dim=config['embed_dim'], 
#             go_term_list=config['go_term_list'], 
#             dropout=config['dropout'], 
#             activation=config['activation'],
#             seed=config['seed'], 
#             add_res=config['add_res'],
#         )
#         # load the state_dict, but only match the keys
#         model.load_state_dict(config['state_dict'], strict=False)
#         return model

