# """ESM-based GO prediction model (baseline)."""

# import torch.nn as nn


# class ESMClassifier(nn.Module):
#     """
#     3-layer MLP classifier on top of ESM embeddings.
#     Serves as a strong baseline for comparison.
#     """
    
#     def __init__(self, num_go_terms: int, esm_dim: int = 1280):
#         super().__init__()
        
#         self.classifier = nn.Sequential(
#             # Layer 1: 1280 → 512
#             nn.Linear(esm_dim, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             # Layer 2: 512 → 512
#             nn.Linear(512, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
            
#             # Layer 3: 512 → num_go_terms
#             nn.Linear(512, num_go_terms)
#         )
    
#     def forward(self, embeddings):
#         """
#         Args:
#             embeddings: (batch, esm_dim) - Pre-computed ESM embeddings
        
#         Returns:
#             logits: (batch, num_go_terms)
#         """
#         return self.classifier(embeddings)