import torch
import torch.nn as nn
import torch.nn.functional as F


def cos_sim(x, y):
    """
    Cosine similarity for a batch of embeddings.
    x: [batch_size, dim]
    y: [batch_size, dim]
    Returns: pairwise similarity [batch_size, batch_size]
    """
    x_norm = x / x.norm(dim=1, keepdim=True)
    y_norm = y / y.norm(dim=1, keepdim=True)
    return torch.matmul(x_norm, y_norm.transpose(0, 1))

def kl_loss(logits_pred, logits_target, tau=1.0):
    """
    KL-divergence loss used for soft-label alignment.
    logits_pred: [batch_size, batch_size] (predicted distribution)
    logits_target: [batch_size, batch_size] (target distribution)
    tau: temperature
    """
    pred_dist = F.softmax(logits_pred / tau, dim=-1)
    target_dist = F.softmax(logits_target / tau, dim=-1)
    loss = F.kl_div(pred_dist.log(), target_dist, reduction='batchmean')
    return loss

###############################################
# The main model class with revised prediction head
###############################################

class AP_align_fuse(nn.Module):
    def __init__(self, tau, hidden_size=1024):
        """
        tau: Temperature for soft-label alignment.
        hidden_size: Hidden size for the intermediate classifier layer.
        """
        super(AP_align_fuse, self).__init__()
        
        self.tau = tau
        
        # Dimensions of the embeddings (adjust as needed)
        self.embedding_dim_seq = 1280
        self.embedding_dim_text = 768
        
        # Additional hyperparameters
        self.num_attr = 17
        self.function_len = 128
        
        self.dropout = nn.Dropout(0.1)
        # If you want a multi-class output, change num_labels accordingly.
        self.num_labels = 3
        
        # Project sequence embeddings (1280-d) to match text dimension (768-d)
        self.project = nn.Linear(self.embedding_dim_seq, self.embedding_dim_text)
        
        # Shared Transformer for aligning both modalities
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dropout=0.1)
        self.share_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Suffix transformer for the sequence branch (token-level)
        seq_suffix_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim_seq, nhead=4, dropout=0.1)
        self.seq_suffix_transformer = nn.TransformerEncoder(seq_suffix_layer, num_layers=2)
        
        # Cross-attention layer to fuse textual info back into the sequence branch
        self.fusion_cross = nn.MultiheadAttention(embed_dim=768, num_heads=4, dropout=0.1, batch_first=True)
        
        # Another transformer for final token-level classification
        token_suffix_layer = nn.TransformerEncoderLayer(d_model=768 + self.embedding_dim_seq, nhead=4, dropout=0.1)
        self.token_suffix_transformer_res = nn.TransformerEncoder(token_suffix_layer, num_layers=2)
        
        # Intermediate classifier layer before the prediction head
        self.fc2_res = nn.Linear(768 + self.embedding_dim_seq, hidden_size)
        self.bn2_res = nn.BatchNorm1d(hidden_size)
        
        # UPDATED: Replace the simple classifier with a multi-layer prediction head
        self.classifier_token = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, self.num_labels)
        )
        
    def forward(self, text_embeddings, seq_embeddings):
        """
        Args:
            text_embeddings: [B, text_len, 768] precomputed text embeddings.
            seq_embeddings: [B, seq_len, 1280] precomputed sequence embeddings.
        Returns:
            Dictionary with:
              - 'token_logits': output predictions as probabilities (after sigmoid)
              - 'contrastive_loss': scalar contrastive loss.
        """
        # 1) Project sequence embeddings to 768-d
        seq_branch_output = self.project(seq_embeddings)  # [B, seq_len, 768]
        
        # 2) Shared transformer alignment for both modalities
        seq_branch_output = self.share_transformer(seq_branch_output)   # [B, seq_len, 768]
        text_branch_output = self.share_transformer(text_embeddings)      # [B, text_len, 768]
        
        # 3) Compute soft-label alignment (contrastive) loss
        cross_modal_loss = self._softlabel_loss_3d(seq_branch_output, text_branch_output, tau=self.tau)
        
        # 4) Fuse textual info back into sequence via cross-attention
        fusion_out, _ = self.fusion_cross(seq_branch_output, text_branch_output, text_branch_output)
        
        # 5) Process the original sequence embeddings with a suffix transformer
        seq_embeddings_suffix = self.seq_suffix_transformer(seq_embeddings)  # [B, seq_len, 1280]
        
        # 6) Concatenate suffix-transformed sequence and fused textual info
        fusion_cat = torch.cat([seq_embeddings_suffix, fusion_out], dim=2)  # [B, seq_len, 1280+768]
        
        # 7) Process with a token-level transformer
        seq_pred = self.token_suffix_transformer_res(fusion_cat)  # [B, seq_len, 1280+768]
        
        # 8) Apply the intermediate classification layer and reshape for batch normalization.
        seq_pred = torch.relu(
            self.bn2_res(self.fc2_res(seq_pred).permute(0,2,1)).permute(0,2,1)
        )
        
        # 9) Pool across the token dimension to get a protein-level representation
        seq_pooled = seq_pred.mean(dim=1)  # [B, hidden_size]
        
        # 10) Pass through the improved prediction head (classifier)
        logits = self.classifier_token(seq_pooled)  # [B, num_labels]
        
        return {
            'token_logits': torch.sigmoid(logits),
            'contrastive_loss': cross_modal_loss,
        }
    
    def _softlabel_loss_3d(self, seq_features, text_features, tau):
        """
        Pools sequence and text features, computes cosine similarities and aligns
        the distributions using a KL divergence.
        """
        seq_mean = seq_features.mean(dim=1)   # [B, 768]
        text_mean = text_features.mean(dim=1)   # [B, 768]
        seq_sim = cos_sim(seq_mean, seq_mean)     # [B, B]
        text_sim = cos_sim(text_mean, text_mean)  # [B, B]
        logits_per_seq, logits_per_text = self._get_similarity(seq_mean, text_mean)
        cross_modal_loss = (kl_loss(logits_per_seq, seq_sim, tau=tau) +
                            kl_loss(logits_per_text, text_sim, tau=tau)) / 2.0
        return cross_modal_loss

    def _get_similarity(self, seq_feats, text_feats):
        """
        Returns pairwise cosine similarity matrices.
        """
        seq_normed = seq_feats / seq_feats.norm(dim=1, keepdim=True)
        text_normed = text_feats / text_feats.norm(dim=1, keepdim=True)
        logits_per_seq = torch.matmul(seq_normed, text_normed.transpose(0,1))
        logits_per_text = logits_per_seq.transpose(0,1)
        return logits_per_seq, logits_per_text

###############################################
# Example usage:
###############################################
if __name__ == "__main__":
    model = AP_align_fuse(0.8, hidden_size=1024)
    
    # Example input dimensions
    seq_emb = torch.randn(52, 1021, 2560)
    txt_emb = torch.randn(52, 128, 768)
    
    out = model(txt_emb, seq_emb)
    print("token_logits shape =", out['token_logits'].shape)
    print("contrastive_loss =", out['contrastive_loss'].item())
