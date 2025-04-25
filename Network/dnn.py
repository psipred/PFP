import torch
import logging
import os
import torch.nn as nn
import torch.optim as optim
# import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Network.model_utils import InterLabelLoss , FmaxMetric
import logging
import os
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool
from ProteinGraphRGCN import EncoderRGCN, EncoderRGCN_GO
from AP_align_fuse import AP_align_fuse
# SimpleDNN(
#   (model): Sequential(
#     (0): Linear(in_features=7680, out_features=512, bias=True)
#     (1): ReLU()
#     (2): Dropout(p=0.3, inplace=False)
#     (3): Linear(in_features=512, out_features=256, bias=True)
#     (4): ReLU()
#     (5): Dropout(p=0.3, inplace=False)
#     (6): Linear(in_features=256, out_features=128, bias=True)
#     (7): ReLU()
#     (8): Dropout(p=0.3, inplace=False)
#     (9): Linear(in_features=128, out_features=32, bias=True)
#   )
# )


# class ProteinGraphRGCN(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, num_relations):
#         super(ProteinGraphRGCN, self).__init__()
#         self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations)
#         self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
#         # We'll do a pooling and then a linear layer to get final embedding of size out_dim
#         self.linear = nn.Linear(hidden_dim, out_dim)
#     def forward(self, x, edge_index, edge_type, batch):
#         # RGCN message passing layers
#         h = F.relu(self.conv1(x, edge_index, edge_type))
#         h = F.relu(self.conv2(h, edge_index, edge_type))
#         # Global pooling (mean) to get graph-level representation
#         hg = global_mean_pool(h, batch)  # batch assigns graph id for each node; here we have a single graph
#         return self.linear(hg)

class SimpleDNNWithAttention(nn.Module):
    def __init__(self, input_dim1, input_dim2, input_dim3, target_dim, hidden_sizes, output_size, go_term_list):
        super(SimpleDNNWithAttention, self).__init__()
        self.go_term_list = go_term_list
        # 将输入特征映射到相同维度
        self.linear1 = nn.Linear(input_dim1, target_dim)  # 用于第一个embedding
        self.linear2 = nn.Linear(input_dim2, target_dim)  # 用于第二个embedding


        # Attention机制的线性变换
        self.attention1 = nn.Linear(target_dim, 1)
        self.attention2 = nn.Linear(target_dim, 1)

        self.bn1 = nn.BatchNorm1d(target_dim)
        self.activation1 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout1 = nn.Dropout(0.5)
        self.linear1_out = nn.Linear(600, 300)



        # DNN
        layers = []
        last_size = input_dim1 + input_dim2 + input_dim3
        print(last_size)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.Dropout(0.3))
            last_size = hidden_size
        
        # 最后一层输出
        layers.append(nn.Linear(last_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, embedding1, embedding2, embedding3):

        # Processing embedding1
        # mapped_embedding1 = self.linear1(embedding1)
        # mapped_embedding1 = self.bn1(mapped_embedding1)
        # mapped_embedding1 = self.activation1(mapped_embedding1)
        # mapped_embedding1 = self.dropout1(mapped_embedding1)
        # mapped_embedding1 = self.linear1_out(mapped_embedding1)

        # Processing embedding2
        # mapped_embedding2 = self.linear2(embedding2)
        # mapped_embedding2 = self.bn1(mapped_embedding2)
        # mapped_embedding2 = self.activation1(mapped_embedding2)
        # mapped_embedding2 = self.dropout1(mapped_embedding2)
        # mapped_embedding2 = self.linear1_out(mapped_embedding2)
        # 将两组特征映射到相同的维度
        # mapped_embedding1 = self.linear1(embedding1)  # shape: (batch_size, target_dim)
        # mapped_embedding2 = self.linear2(embedding2)  # shape: (batch_size, target_dim)

        # # 计算注意力权重
        # weight1 = self.attention1(mapped_embedding1)  # shape: (batch_size, 1)
        # weight2 = self.attention2(mapped_embedding2)  # shape: (batch_size, 1)
        

        # attention_weights = F.softmax(torch.cat([weight1, weight2], dim=1), dim=1)  # shape: (batch_size, 2)


        # # 根据注意力权重对特征进行调整
        # adjusted_embedding1 = attention_weights[:, 0].unsqueeze(1) * mapped_embedding1  # (batch_size, target_dim)
        # adjusted_embedding2 = attention_weights[:, 1].unsqueeze(1) * mapped_embedding2  # (batch_size, target_dim)

        # 注意力拼接
        # fused_embedding = torch.cat([adjusted_embedding1, adjusted_embedding2], dim=1)  # (batch_size, 2 * target_dim)
        # 动态融合特征
        # fused_embedding = (
        #     attention_weights[:, 0].unsqueeze(1) * mapped_embedding1 +
        #     attention_weights[:, 1].unsqueeze(1) * mapped_embedding2
        # )  # shape: (batch_size, target_dim)



        # fused_embedding = (
        #     mapped_embedding1 +
        #     mapped_embedding2
        # )
        # fused_embedding =  torch.cat([mapped_embedding1, mapped_embedding2], dim=1)

        fused_embedding =  torch.cat([embedding1, embedding2, embedding3], dim=1)

        
        # exit(fused_embedding.shape)
        # 融合后的特征输入DNN模型
        output = self.model(fused_embedding)
        return output

class SimpleDNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, go_term_list):
        super(SimpleDNN, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:

            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # Add Batch Normalization
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.Dropout(0.4))  # Dropout for regularization
            last_size = hidden_size

        layers.append(nn.Linear(last_size, output_size))  # Final output layer
        self.model = nn.Sequential(*layers)
        self.prediction_mode = False
        self.go_term_list = go_term_list

        # initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        

    def forward(self, x):

        y_pred = self.model(x)
        if self.prediction_mode:
            y_pred = torch.sigmoid(y_pred)
        return y_pred
    
    @staticmethod
    def load_config(save_path:str):
        config = torch.load(save_path, map_location=torch.device('cpu'))
        model = SimpleDNN(
            input_size=config['input_size'], 
            hidden_sizes=config['hidden_sizes'], 
            output_size=config['output_size'], 
        )
        # load the state_dict, but only match the keys
        model.load_state_dict(config['state_dict'], strict=False)
        return model



class SimpleTrainer:
    def __init__(self, model_params, weight_matrix, go_term_list, child_matrix, device="cuda"):
        """Initialize model, loss function, optimizer, and other training parameters."""


        self.tau = 0.8
        self.model_params = model_params

        if model_params['embedding_type'] == 'GCRN': 

                # Initialize the RGCN model
            in_dim = 26        # input feature dimension (size of node feature vector)
            hidden_dim = 64
            out_dim = 64                     # dimension of graph embedding we want
            num_relations = 3  # 3 in our case
            self.model = EncoderRGCN_GO(input_dim=26, hidden_dim=512, n_layers=3, emb_dim=512, num_relations=3, num_go_terms = model_params['output_size'])
            self.seqmodel = SimpleDNN(model_params['input_size'], model_params['hidden_sizes'], model_params['output_size'], go_term_list)
            # Create the combined model
            self.model = CombinedGOModel(self.model, self.seqmodel)
            # graph_emb = model(data.x, data.edge_index, data.edge_type, batch=torch.zeros(data.num_nodes, dtype=torch.long))
            # graph_emb = graph_emb.detach().numpy()
            # print(data.x.shape)
            # print(data.edge_index.shape)
            # print(graph_emb.shape)
        elif model_params['embedding_type'] == 'mmstie':


            self.model = AP_align_fuse(self.tau, hidden_size=256)


            old_weights = torch.load("testdata/best_model_fuse_0.8322829131652661.pt", map_location=torch.device('cuda'))

            # load_state_dict(..., strict=False) means it will ignore shape mismatches
            # or layers not present in the new model.
            # Safely remove keys if they exist
            old_weights.pop("classifier_token.weight", None)
            old_weights.pop("classifier_token.bias", None)
            missing_keys, unexpected_keys = self.model.load_state_dict(old_weights, strict=False)
            
            
            print("Missing keys in loaded state:", missing_keys)
            # print("Unexpected keys in loaded state:", unexpected_keys)




            # 1) Freeze all parameters in the model
            for _, param in self.model.named_parameters():
                param.requires_grad = False

            # 2) Unfreeze just the classifier (or any layer you still want to train)
            for _, param in self.model.classifier_token.named_parameters():
                param.requires_grad = True

            # exit()


        elif model_params['embedding_type'] == 'attention': 
            self.model = SimpleDNNWithAttention(1024, 7680, 1536, 1024, model_params['hidden_sizes'], model_params['output_size'], go_term_list)

        else:

            self.model = SimpleDNN(model_params['input_size'], model_params['hidden_sizes'], model_params['output_size'], go_term_list)

        # exit()
        # exit()


        print(self.model)
        # print(self.seqmodel)



        self.criterion = nn.BCEWithLogitsLoss()
        # elif model_params['loss_function'] == 'inter':
            # self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate,  weight_decay=1e-5)
        
        
        self.optimizer = optim.AdamW(self.model.parameters(), model_params['lr'])  # Use AdamW optimizer

        
        # Training parameters
        self.num_epochs = model_params['num_epochs']
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        # Move model to the specified device
        self.model = self.model.to(self.device)

        # exit(self.model.device)
        # Metrics storage
        self.Mtrain_losses = []
        self.Mval_losses = []
        self.Mval_f1_scores = []
        self.Mval_precisions = []
        self.Mval_recalls = []

        self.Mval_f1_scores50 = []
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.val_precisions = []
        self.val_recalls = []

        self.weight_matrix = weight_matrix
        self.loss_fn = InterLabelLoss(device=self.device)

        # self.child_matrix = child_matrix

        self.metrics = FmaxMetric(device=self.device, weight_matrix=weight_matrix, child_matrix=child_matrix)

        
        

        self.logger = ExperimentLogger(
            log_dir = f"Experiments/{model_params['task']}/{model_params['loss_function']}_Loss_Log/{model_params['embedding_type']}",

            log_file=f"{model_params['embedding_type']} Embedding"
        )

        # 记录实验参数
        self.experiment_args = model_params
        self.logger.log_experiment_info(args=self.experiment_args)

    def set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def train(self, train_loader, val_loader=None):
        """Train the model using the provided DataLoader."""
        self.logger.log_message("Starting experiment...", level="info")
        self.set_seed(42)  # set seed

        for epoch in range(self.num_epochs):
            self.logger.log_message(f"{'-'*20} {epoch + 1} {'-'*20}", level="info")
            self.model.train()  # Set to training mode
            running_loss = 0.0

            # for batch_idx, (names, inputs, y_true) in tqdm(enumerate(train_loader), total=len(train_loader), ascii=' >='):
           
            # for batch_idx, (names, data, data2, labels) in tqdm(enumerate(train_loader), total=len(train_loader), ascii=' >='):


            for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), ascii=' >='):


                if False:
                    names, seq,  graph, labels = batch

                    graph, seq, labels = graph.to(self.device), seq.to(self.device), labels.to(self.device)
                    

                    # # for graph
                    # outputs_G = self.model(x=graph.x, 
                    #       edge_index=graph.edge_index, 
                    #       edge_type=graph.edge_type, 
                    #       batch=graph.batch)
                    # outputs_S = self.seqmodel(seq)
                    # print(outputs_S)
                    # print(outputs_G.size())
                    # exit(outputs_S.size())

                    outputs =  self.model(graph_x=graph.x,edge_index=graph.edge_index, edge_type=graph.edge_type, batch=graph.batch, seq_x = seq)
                    # exit(outputs)
                    # exit(outputs.size)
                elif len(batch) == 5:

                    names, data, data2, data3, labels = batch
                    data, data2, data3, labels = data.to(self.device), data2.to(self.device), data3.to(self.device), labels.to(self.device)
                    outputs = self.model(data, data2, data3)

                
                elif len(batch) == 4:
                    names, data, data2, labels = batch
                    data, data2, labels = data.to(self.device), data2.to(self.device), labels.to(self.device)
                    # print(data2.size())
                    # exit(data.size())
                    outputs = self.model(data2, data)


                    logits = outputs["token_logits"]
                    contrastive_loss = outputs["contrastive_loss"]

                    print("contrastive_loss", str(contrastive_loss))
                    outputs = logits

                    
                    # Process the batch with four elements
                elif len(batch) == 3:
                    



                    # # below is origin
                    names, seq,  graph, labels = batch

                    # exit(data)
                    # exit(data.shape)
                    # print(data)
                    # print(data.edge_index)
                    # print(data.edge_type)
                    # exit()
                    data, labels = data.to(self.device), labels.to(self.device)
                    

                    # for graph
                    outputs = self.model(x=data.x, 
                          edge_index=data.edge_index, 
                          edge_type=data.edge_type, 
                          batch=data.batch)


                    # the aboveeeeeee is origin

                    
                    # outputs = self.model(data)





                    # Process the batch with three elements

                # print(data.shape)
                # print(data2.shape)



                # Forward pass
                # torch.Size([512, 1318])

                
                



                
                if self.model_params['loss_function'] == 'cross_entropy':
                    loss = self.criterion(outputs, labels)


                elif self.model_params['loss_function'] == 'inter':
                    # exit(outputs.size())

                    loss = self.loss_fn(outputs, labels, weight_tensor = self.weight_matrix, aspect="BPO", print_loss=False)


                # Backward and optimize                                     
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                # print("  Current layer weight:\n", self.model.fusion.alpha)
            
            # Calculate average training loss for the epoch
            avg_train_loss = running_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            
            print(f"Epoch [{epoch+1}/{self.num_epochs}] Training Loss: {avg_train_loss:.4f}")
            
            # Validation
            if isinstance(val_loader, torch.utils.data.DataLoader):  # Ensure val_loader is a DataLoader
                val_loss, val_precision, val_recall, val_f1, mean_precision, mean_recall, mean_f1_score, f150 = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_precisions.append(val_precision)
                self.val_recalls.append(val_recall)
                self.val_f1_scores.append(val_f1)



                val_metrics = {
                    "Average Precision": val_precision,
                    "Weighted Precision": mean_precision,
                    "Average Recall": val_recall,
                    "Weighted Recall": mean_recall,
                    "Average F1 Score": val_f1,
                    "Weighted F1 Score": mean_f1_score,
                    "Weighted F1 Score Cutoff 0.5": f150
                }

                
                self.logger.log_epoch_metrics(epoch + 1 , avg_train_loss, val_loss, val_metrics)
                
                print(f"Epoch [{epoch+1}/{self.num_epochs}] Validation Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}")

            else:
                print("Warning: val_loader is not a DataLoader. Skipping validation.")
        
        
        self.logger.save_model(self.model, epoch + 1)
        print("Training complete!")
        exit()
        self.plot_metrics(f"Experiments/{self.model_params['task']}/{self.model_params['loss_function']}_Loss_Log/{self.model_params['embedding_type']}/plot")  # Plot metrics after training
        # exit()

    def validate(self, val_loader):
        """Evaluate the model on the validation dataset."""
        self.model.eval()  # Set to evaluation mode
        val_loss = 0.0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_samples = 0


        y_preds = []
        y_trues = []
        

        
        with torch.no_grad():  # Disable gradients for validation
            # for names, data, data2, labels in val_loader:
            #     data, data2, labels = data.to(self.device), data2.to(self.device), labels.to(self.device)

                
            for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), ascii=' >='):



                if False:
                    names, seq,  graph, labels = batch

                    graph, seq, labels = graph.to(self.device), seq.to(self.device), labels.to(self.device)
                    

                    # # for graph
                    # outputs_G = self.model(x=graph.x, 
                    #       edge_index=graph.edge_index, 
                    #       edge_type=graph.edge_type, 
                    #       batch=graph.batch)
                    # outputs_S = self.seqmodel(seq)
                    # print(outputs_S)
                    # print(outputs_G.size())
                    # exit(outputs_S.size())

                    outputs =  self.model(graph_x=graph.x,edge_index=graph.edge_index, edge_type=graph.edge_type, batch=graph.batch, seq_x = seq)

                elif len(batch) == 5:
                    names, data, data2, data3, labels = batch
                    data, data2, data3, labels = data.to(self.device), data2.to(self.device), data3.to(self.device), labels.to(self.device)
                    outputs = self.model(data, data2, data3)
                elif len(batch) == 4:
                    # names, data, data2, labels = batch
                    # data, data2, labels = data.to(self.device), data2.to(self.device), labels.to(self.device)
                    # outputs = self.model(data, data2)
                    # Process the batch with four elements

                    names, data, data2, labels = batch
                    data, data2, labels = data.to(self.device), data2.to(self.device), labels.to(self.device)
                    # print(data2.size())
                    # exit(data.size())
                    outputs = self.model(data2, data)


                    logits = outputs["token_logits"]
                    contrastive_loss = outputs["contrastive_loss"]

                    print("contrastive_loss", str(contrastive_loss))
                    outputs = logits

                elif len(batch) == 3:
                    names, data, labels = batch
                    data, labels = data.to(self.device), labels.to(self.device)






                    outputs = self.model(x=data.x, 
                          edge_index=data.edge_index, 
                          edge_type=data.edge_type, 
                          batch=data.batch)
                    # outputs = self.model(data)



                if self.model_params['loss_function'] == 'cross_entropy':
                    loss = self.criterion(outputs, labels)
                elif self.model_params['loss_function'] == 'inter':
                    loss = self.loss_fn(outputs, labels, weight_tensor = self.weight_matrix, aspect="BPO", print_loss=False)

                # Forward pass
                # outputs = self.model(data, data2)
                
                # loss = self.criterion(outputs, labels)
                # loss = self.loss_fn(outputs, labels, weight_tensor = self.weight_matrix, aspect="BPO", print_loss=False)
                val_loss += loss.item()

                # Threshold predictions at 0.5
                predicted = (torch.sigmoid(outputs) > 0.5).float()


                y_preds.append(torch.sigmoid(outputs))
                y_trues.append(labels)
                # Calculate true positives, false positives, and false negatives
                true_positives += (predicted * labels).sum().item()
                false_positives += (predicted * (1 - labels)).sum().item()
                false_negatives += ((1 - predicted) * labels).sum().item()
                total_samples += labels.numel()

                

                # print(y_preds)

                # exit(y_trues.shape, y_trues)

                # After collecting predictions and true labels in a list:
                # y_preds = torch.cat(y_preds, dim=0)  # Combine list of tensors into one tensor
                

                # print(y_preds)
                # exit()

        # Concatenate all tensors from the list into one tensor along the first dimension
        y_preds = torch.cat(y_preds, dim=0).cuda()
        y_trues = torch.cat(y_trues, dim=0).cuda()
        mean_f1_score, mean_precision, mean_recall, cut_off, f150 = self.metrics.compute_protein_centric_fm(y_preds, y_trues, margin=0.01, weight_tensor=self.weight_matrix)
        mean_f1_score = mean_f1_score.item()
        mean_precision = mean_precision.item()
        mean_recall = mean_recall.item()
        cut_off = cut_off.item()
        # round the cut_off to 3 decimal places
        cut_off = round(cut_off, 3)
        # round other to 5 decimal places
        mean_f1_score = round(mean_f1_score, 5)
        mean_precision = round(mean_precision, 5)
        mean_recall = round(mean_recall, 5)

        
        print("f150", f150, "mean_f1 score: " , mean_f1_score, "mean_precision", mean_precision, "mean_recall", mean_recall, "cut_off", cut_off)

        self.Mval_f1_scores50.append(f150)
        self.Mval_f1_scores.append(mean_f1_score)
        self.Mval_precisions.append(mean_precision)
        self.Mval_recalls.append(mean_recall)







        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

        # y_preds = torch.cat(y_preds, dim=0).cpu()
        # y_trues = torch.cat(y_trues, dim=0)
        
        # exit()

        return avg_val_loss, precision, recall, f1_score, mean_precision, mean_recall, mean_f1_score, f150

    def plot_metrics(self, path):
        """Plot training and validation metrics with grouped comparisons."""
        epochs = range(1, self.num_epochs + 1)
        # Set up the figure size
        plt.figure(figsize=(15, 12))

        # Plot training and validation loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss', color='blue')
        plt.plot(epochs, self.val_losses, label='Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot mean and weighted precision
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.Mval_precisions, label='Weighted Validation Precision', color='green')
        plt.plot(epochs, self.val_precisions, label='Average Validation Precision', color='purple')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.title('Average vs Weighted Validation Precision')
        plt.legend()

        # Plot mean and weighted recall
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.Mval_recalls, label='Weighted Validation Recall', color='red')
        plt.plot(epochs, self.val_recalls, label='Average Validation Recall', color='cyan')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.title('Average vs Weighted Validation Recall')
        plt.legend()

        # Plot weighted, average, and F1@50
        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.Mval_f1_scores, label='Weighted F1 Score (wFmax)', color='teal')
        plt.plot(epochs, self.val_f1_scores, label='Average F1 Score', color='darkorange')
        plt.plot(epochs, self.Mval_f1_scores50, label='F1@50 (Cutoff 0.5)', color='gold', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title('Weighted vs Average vs F1@50')
        plt.legend()

        plt.tight_layout()
        print(path)
        plt.savefig(path)
        plt.show()
    

    




class ExperimentLogger:
    def __init__(self, log_dir="logs", log_file="experiment.log"):
        """初始化日志记录器"""


        self.ensure_clean_file(log_dir, log_file)
        # if os.path.exists(log_dir):
        #     shutil.rmtree(log_dir)  # 删除整个目录及其内容
        # os.makedirs(log_dir, exist_ok=True)  # 重新创建目录

        log_path = os.path.join(log_dir, log_file)
        
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        
        self.logger = logging.getLogger()
        self.log_dir = log_dir
        self.log_file = log_file

    def log_experiment_info(self, args=None, metrics=None):
        """记录实验参数和结果"""
        if args:
            self.logger.info("Experiment Parameters:")
            for key, value in args.items():
                self.logger.info(f"{key}: {value}")
        
        if metrics:
            self.logger.info("Metrics:")
            for key, value in metrics.items():
                self.logger.info(f"{key}: {value}")
    
    def log_message(self, message, level="info"):
        """记录普通消息"""
        log_func = {
            "info": self.logger.info,
            "debug": self.logger.debug,
            "warning": self.logger.warning,
            "error": self.logger.error,
            "critical": self.logger.critical,
        }.get(level, self.logger.info)
        log_func(message)

    # def save_model(self, model, epoch):
    #     """保存模型权重"""
    #     model_path = os.path.join(self.log_dir, f"{self.log_file}_model_epoch_{epoch}.pth")
    #     torch.save(model.state_dict(), model_path)
    #     self.logger.info(f"Model saved at: {model_path}")

    def save_model(self, model, epoch):
        """Save the entire model."""
        model_path = os.path.join(self.log_dir, f"{self.log_file}_model_epoch_{epoch}.pt")
        torch.save(model, model_path)
        self.logger.info(f"Entire model saved at: {model_path}")


        # 'go_term_list': self.go_term_list, 

    # def save_config(self, model, epoch):
    #     save_path = os.path.join(self.log_dir, f"{self.log_file}_model_epoch_{epoch}.pt")
    #     config = {
    #         'aspect': self.aspect, 
    #         'layer_list': self.layer_list, 
    #         'embed_dim': self.embed_dim, 
    #         'go_term_list': self.go_term_list, 
    #         'dropout': self.dropout, 
    #         'activation': self.activation,
    #         'seed': self.seed, 
    #         'add_res': self.add_res,
    #         'state_dict': self.state_dict(),
    #     }
    #     torch.save(config, save_path)





    def log_epoch_metrics(self, epoch, train_loss, val_loss=None, val_metrics=None):
        """记录每个 Epoch 的训练和验证结果"""
        self.logger.info(f"Epoch [{epoch}] Training Loss: {train_loss:.4f}")
        if val_loss is not None and val_metrics:
            self.logger.info(f"Epoch [{epoch}] Validation Loss: {val_loss:.4f}")
            for metric, value in val_metrics.items():
                self.logger.info(f"Validation {metric}: {value:.4f}")


    
    
    def ensure_clean_file(self, log_dir, log_file):
        """
        确保只覆盖指定文件，而不删除整个文件夹
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)  # 如果文件夹不存在，直接创建
        else:
            file_path = os.path.join(log_dir, log_file)
            if os.path.exists(file_path):
                os.remove(file_path)  # 删除指定文件




class AdaptiveFusion(nn.Module):
    """
    Learns weights to fuse two modality outputs (graph and sequence).
    The weights are learned as parameters and normalized using softmax.
    """
    def __init__(self):
        super(AdaptiveFusion, self).__init__()
        # Initialize two scalar weights, one per branch
        self.alpha = nn.Parameter(torch.tensor([0.2, 0.8]))

    def forward(self, feat_graph, feat_seq):
        # Normalize the weights so they sum to 1
        weights = torch.softmax(self.alpha, dim=0)
        # Fuse features via weighted sum
        fused = weights[0] * feat_graph + weights[1] * feat_seq
        return fused

class CombinedGOModel(nn.Module):
    def __init__(self, graph_model, seq_model):
        super(CombinedGOModel, self).__init__()
        self.graph_model = graph_model
        self.seq_model = seq_model
        self.fusion = AdaptiveFusion()
        # Optionally, you can add a final projection or activation here

    def forward(self, graph_x, edge_index, edge_type, batch, seq_x):
        # Forward pass for graph branch
        graph_logits = self.graph_model(graph_x, edge_index, edge_type, batch)  # shape: [batch, num_go_terms]

        # print(graph_logits)
        # Forward pass for sequence branch
        seq_logits = self.seq_model(seq_x)  # shape: [batch, num_go_terms]
        # print(seq_logits)

        # Adaptive fusion of both outputs
        fused_logits = self.fusion(graph_logits, seq_logits)
        return fused_logits
