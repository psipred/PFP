import os, random, pickle
import numpy as np
import torch
import scipy.sparse as ssp
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import List, Union
import multiprocessing as mp
from tqdm import tqdm
from sklearn.decomposition import PCA
import torch.nn.functional as F
import math


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class InterlabelGODataset(Dataset):
    def __init__(self,
        features_dir:str,
        embedding_type:str,
        names_npy:str,
        labels_npy:str = None,
        repr_layers:list = [34, 35, 36],
        # repr_layers:list=[36],
        low_memory:bool = False,
        combine_feature:bool = False,
        pca:bool=True,
        do_normalize:bool=True, 
        
    ):

        # print("features_dir", features_dir)
        # exit()
        self.features_dir = features_dir
        self.names_npy = names_npy
        self.repr_layers = repr_layers

        self.low_memory = low_memory
        self.feature_cache = dict()
        self.temp = dict()
        self.ankh = dict()
        self.embedding_type = embedding_type

        self.combine_feature = combine_feature
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


        # print("dgdsgsdgdsg", self.names)
        if not self.prediction:

            self.labels = self.load_labels(self.labels_npy)

        
        if self.combine_feature and not self.embedding_type == "GCRN":

            if  self.embedding_type == 'mmstie':

                for name in tqdm(self.names):
                    # seq, text
                    self.feature_cache[name], self.temp[name] = self.load_featureMM(name)
                    # exit()
                    
            else: 
                for name in tqdm(self.names): 
                        self.feature_cache[name] = self.load_featureT5(name)
                        self.ankh[name] = self.load_featureAnhk(name)
                        self.temp[name] = self.load_feature(name)


                if pca:
                    assert self.feature_cache.keys() == self.temp.keys(), "Keys Donot Match！"

                    # # To matrix
                    # names = list(self.feature_cache.keys())  # sample name list
                    # # embedding_1024 = np.array([self.feature_cache[name] for name in names])  # shape (n_samples, 1024)
                    # embedding_2048 = np.array([self.temp[name] for name in names])  # shape (n_samples, 2048)

                    # # dimension size
                    # target_dim = 1024

                    # pca_2048 = PCA(n_components=target_dim)
                    # reduced_2048 = pca_2048.fit_transform(embedding_2048)

                    # self.temp = {name: reduced_2048[i] for i, name in enumerate(names)}



                    # # 假设所有嵌入向量均为 1024 维
                    # embedding_dim = 1024
                    # attention_fusion = AttentionFusion(embedding_dim)
                    # # 将字典转换为张量
                    # feature1 = torch.stack([torch.tensor(self.feature_cache[name]) for name in names])
                    # feature2 = torch.stack([torch.tensor(self.temp[name]) for name in names])

                    # # 融合
                    # fused_features = attention_fusion(feature1, feature2)

                    # # 生成融合后的字典
                    # self.fused_dict = {name: fused_features[i].detach().numpy() for i, name in enumerate(names)}


        else:




            if not self.low_memory and self.embedding_type == "GCRN":
                self.features_dir = 'new_embedding/structure/PyG_normalized'
                filtered_names = []
                    

                    
                for n in self.names:
                    path = os.path.join(self.features_dir, n + '.pt')
                    if os.path.exists(path):
                        filtered_names.append(n)
                    else:
                        # print(f"Skipping {n}, file missing: {path}")
                        pass


                self.names = filtered_names


                for name in tqdm(self.names):
                    # exit(name)
                    self.feature_cache[name] = self.load_feature(name)
                    self.temp[name] = self.load_structure(name)


            elif not self.low_memory and self.embedding_type == "T5":

                print("T5")
                for name in tqdm(self.names):
                    self.feature_cache[name] = self.load_featureT5(name)


            elif not self.low_memory and self.embedding_type == "ESM":
                print("ESM")
                for name in tqdm(self.names):
                    self.feature_cache[name] = self.load_feature(name)
                    # exit(self.feature_cache[name].shape)
            elif not self.low_memory and self.embedding_type == "Ahkh":                
                for name in tqdm(self.names):                
                    self.feature_cache[name] = self.load_featureAnhk(name)

            



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


        if name not in self.feature_cache and not self.low_memory:
            feature = self.load_feature(name)
            self.feature_cache[name] = feature

        
        elif self.combine_feature:
            # feature = self.fused_dict[name]
            if self.embedding_type == "GCRN":
                if self.prediction:
                    return name, self.feature_cache[name], self.temp[name]
                # return name, self.feature_cache[name], self.temp[name], self.ankh[name], self.labels[idx]
                # print(name, self.feature_cache[name], self.temp[name], self.labels[idx])
                # exit()

                return {
                        "name":name,
                        "seq": self.feature_cache[name],
                        "graph": self.temp[name],
                        "label":  self.labels[idx]
                }
                # name, self.feature_cache[name], self.temp[name], self.labels[idx]
            elif self.embedding_type == "mmstie":
                # print(self.feature_cache[name].size())
                # print(self.temp[name].size())
                # exit()
                # seq, text
                return name, self.feature_cache[name], self.temp[name], self.labels[idx]

            else:
                if self.prediction:
                    return name, self.feature_cache[name], self.temp[name], self.ankh[name]
                # return name, self.feature_cache[name], self.temp[name], self.ankh[name], self.labels[idx]
                
                return name, self.feature_cache[name], self.temp[name], self.ankh[name], self.labels[idx]
                # feature = np.concatenate([self.feature_cache[name], self.temp[name], self.ankh[name]], axis=-1)
        else:

            
            feature = self.feature_cache[name]

            


            
        if self.prediction:
            return name, feature
        else:
            label = self.labels[idx]

            return name, feature, label
        
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

        self.features_dir = 'testdata'


        

        features_seq = np.load(os.path.join(self.features_dir, 'embedding', name + '.npy'),allow_pickle=True).item()
        features_text = np.load(os.path.join(self.features_dir,'biomed_text_embeddingss', name + '.npy'),allow_pickle=True).item()
        # print(features_seq["per_tok"][34].size())
        # print(features_text.size())

    
        return torch.from_numpy(features_seq["embedding"]).float(), torch.from_numpy(features_text["embedding"]).float()

    
    def load_structure(self, name:str)->np.ndarray:
        #  
        # #  name =  'A0A0B4J1F4'
        #  features = torch.load(os.path.join(self.features_dir, name + '.pt'), weights_only=False)
        #  exit(features)

            self.features_dir = 'new_embedding/structure/PyG_normalized'
            path = os.path.join(self.features_dir, name + '.pt')
            # 1) Check if file exists
            if not os.path.exists(path):
                print(f"Warning: {path} not found!")
                return None
            # 2) Try loading
            try:
                loaded_data = torch.load(path, weights_only=False)  # should be a PyG Data object
                # loaded_data.x is your node feature matrix, etc.
                return loaded_data
            except Exception as e:
                print(f"Error loading {path}: {e}")
                return None




        #  return torch.from_numpy(features).float()
    

    def load_featureAnhk(self, name:str)->np.ndarray:
         self.features_dir = 'new_embedding/ankh'
         features = np.load(os.path.join(self.features_dir, name + '.npy'),allow_pickle=True)

         return torch.from_numpy(features).float()


    def load_featureT5(self, name:str)->np.ndarray:
        """
        Load feature from npy file.

        Args:
            name (str): name of the feature

        Returns:
            np.ndarray: feature
        """
        # train
        self.features_dir = 'new_embedding/t5'

        # test
        self.features_dir = '/Users/zjzhou/Downloads/InterLabelGO+/example/T5_test_emb'



        
        features = np.load(os.path.join(self.features_dir, name + '.npy'),allow_pickle=True)


        # random_feature = np.random.choice([0, 1], size=1024)
        # features = random_feature


        return torch.from_numpy(features).float()



    def load_feature(self, name:str)->np.ndarray:
        """
        Load feature from npy file.

        Args:
            name (str): name of the feature

        Returns:
            np.ndarray: feature
        """
        #training 
        self.features_dir = 'Data/embeddings'
        # test
        # self.features_dir = '/Users/zjzhou/Downloads/InterLabelGO+/test_embedding'
        # exit(os.path.join(self.features_dir, name + '.npy'))
        # print(os.path.join(self.features_dir, name + '.npy'))
        features = np.load(os.path.join(self.features_dir, name + '.npy'),allow_pickle=True).item()
        # print('finish')
        final_features = []
        self.repr_layers = [34, 35, 36]
        for layer in self.repr_layers:

            final_features.append(features['mean'][layer])
        # exit(final_features)
        final_features = np.concatenate(final_features)

        # random_feature = np.random.choice([-1, 1], size=2560)
        # exit(final_features.shape)
        # final_features = random_feature
        return torch.from_numpy(final_features).float()



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

class InterlabelGODatasetWindow(Dataset):
    def __init__(self,
        features_dir:str,
        fasta_dict:dict,
        repr_layers:list=[34, 35, 36],
        window_size:int=50,
    ):
        self.features_dir = features_dir
        self.fasta_dict = fasta_dict
        self.repr_layers = repr_layers
        self.window_size = window_size
        self.data_list = self.load_data()

    def load_data(self):
        data_dict = {}
        for name, seq in self.fasta_dict.items():
            # truncate the sequence to the first 1000 amino acids, because the esm model only accept 1000 amino acids
            if len(seq) > 1000:
                seq = seq[:1000]
            features = np.load(os.path.join(self.features_dir, name + '.npy'),allow_pickle=True).item()
            seq_len = len(seq)
            # create windows
            for i in range(0, seq_len-self.window_size+1):
                start = i
                end = i + self.window_size
                final_features = []
                for layer in self.repr_layers:
                    final_features.append(features['per_tok'][layer][start:end].mean(axis=0))
                final_features = np.concatenate(final_features)
                final_features = torch.from_numpy(final_features).float()
                data_dict[name + '_' + str(start) + '-' + str(end)] = final_features
        return list(data_dict.items())

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        name, feature = self.data_list[idx]
        return name, feature

class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class InterLabelResNet(nn.Module):
    def __init__(self, 
        aspect:str=None, # aspect of the GO terms
        layer_list:list=[1024], # layers of dnn network, example [512, 256, 128]
        embed_dim:int=2560, # dim of the embedding protein language model
        go_term_list:List[str]=[], # list of GO terms for prediction
        dropout:float=0.3, # dropout rate
        activation:str='elu', # activation function
        seed:int=42, # random seed
        prediction_mode:bool=False, # if True, the model will output the prediction of the GO terms
        add_res:bool=False,
        ):
        super(InterLabelResNet, self).__init__()
        self.aspect = aspect
        self.layer_list = layer_list
        self.embed_dim = embed_dim
        self.go_term_list = go_term_list
        self.vec2go_dict = self.get_vec2go()
        self.class_num = len(go_term_list)
        self.dropout = dropout
        self.activation = activation
        self.seed = seed
        self.prediction_mode = prediction_mode
        self.add_res = add_res

        # bach normalization for the input
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.bn3 = nn.BatchNorm1d(embed_dim)

        # Define DNN branches
        self.branch1 = self._build_dnn_branch(embed_dim)
        self.branch2 = self._build_dnn_branch(embed_dim)
        self.branch3 = self._build_dnn_branch(embed_dim)
        
        # concat dense layer
        self.concat_layer = nn.Sequential(
            nn.Linear(layer_list[-1]*3, (layer_list[-1])),
            self.get_activation(activation),
            nn.Dropout(dropout),
            nn.BatchNorm1d((layer_list[-1])),
        )


        if self.add_res:
            self.res = Residual(
                nn.Sequential(
                    nn.Linear(layer_list[-1], layer_list[-1]),
                    self.get_activation(activation),
                    nn.Dropout(0.1),
                    nn.BatchNorm1d((layer_list[-1])),
                )
            )

        self.output_layer = nn.Sequential(
            nn.Linear((layer_list[-1]), self.class_num),
            #nn.Sigmoid(),
        )

        # initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def _build_dnn_branch(self, embed_dim):
        layers = []
        for i, layer in enumerate(self.layer_list):
            layers.append(nn.Linear(embed_dim if i == 0 else self.layer_list[i - 1], layer))
            layers.append(self.get_activation(self.activation))
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.BatchNorm1d(layer))
        return nn.Sequential(*layers) # * is used to unpack the list for the nn.Sequential

    def forward(self, inputs):
        # print(self.embed_dim)
        # print(inputs.shape)
        x1 = inputs[:, :self.embed_dim]
        x2 = inputs[:, self.embed_dim:2*self.embed_dim]
        x3 = inputs[:, 2*self.embed_dim:]


        # batch normalization for each branch
        x1 = self.bn1(x1)
        x2 = self.bn2(x2)
        x3 = self.bn3(x3)
        # print(inputs.shape)
        # print(x1.shape, x2.shape, x3.shape)
        # 
        
        x1 = self.branch1(x1)

        x2 = self.branch2(x2)
        x3 = self.branch3(x3)




        x = torch.cat((x1, x2, x3), dim=1)
        x = self.concat_layer(x)

        if self.add_res:
            x = self.res(x)

        y_pred = self.output_layer(x)

        if self.prediction_mode:
            y_pred = torch.sigmoid(y_pred)

        return y_pred
    
    def get_vec2go(self):
        vec2go_dict = dict()
        for i, go_term in enumerate(self.go_term_list):
            vec2go_dict[i] = go_term
        return vec2go_dict
    
    def get_activation(self, activation:str):
        activation = activation.lower()
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leakyrelu':
            return nn.LeakyReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError('activation function not supported')
    
    def save_config(self, save_path:str):
        config = {
            'aspect': self.aspect, 
            'layer_list': self.layer_list, 
            'embed_dim': self.embed_dim, 
            'go_term_list': self.go_term_list, 
            'dropout': self.dropout, 
            'activation': self.activation,
            'seed': self.seed, 
            'add_res': self.add_res,
            'state_dict': self.state_dict(),
        }
        torch.save(config, save_path)
        

    @staticmethod
    def load_config(save_path:str):
        config = torch.load(save_path, map_location=torch.device('cpu'))
        model = InterLabelResNet(
            aspect=config['aspect'], 
            layer_list=config['layer_list'], 
            embed_dim=config['embed_dim'], 
            go_term_list=config['go_term_list'], 
            dropout=config['dropout'], 
            activation=config['activation'],
            seed=config['seed'], 
            add_res=config['add_res'],
        )
        # load the state_dict, but only match the keys
        model.load_state_dict(config['state_dict'], strict=False)
        return model

