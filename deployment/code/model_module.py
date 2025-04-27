import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
class Config:
    EMBEDDING_DIM = 32
    TRANSFORMER_LAYERS = 2
    ATTENTION_HEADS = 4
    DROPOUT = 0.1

class TowerBlock(nn.Module):
    def __init__(self, num_features, vocab_dict, bin_features, emb_dim=Config.EMBEDDING_DIM):
        super().__init__()
        # numeric feature processing
        self.num_proj = nn.Sequential(
            nn.Linear(num_features, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT)
        )
        
        # categorical embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(size, emb_dim)
            for name, size in vocab_dict.items()
        })
        
        # binary features projection
        self.bin_proj = nn.Sequential(
            nn.Linear(bin_features, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT)
        )
        
        # feature interaction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=Config.ATTENTION_HEADS,
            dim_feedforward=emb_dim * 4,
            dropout=Config.DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=Config.TRANSFORMER_LAYERS
        )
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(emb_dim * 2, emb_dim)
        )

    def forward(self, num, cat, bin):
        # Process numeric features
        num_emb = self.num_proj(num).unsqueeze(1)
        
        # process categorical features
        cat_embs = []
        for i, (name, emb) in enumerate(self.embeddings.items()):
            cat_embs.append(emb(cat[:, i]))
        cat_emb = torch.stack(cat_embs, dim=1)
        
        # process binary features
        bin_emb = self.bin_proj(bin).unsqueeze(1)
        
        x = torch.cat([num_emb, cat_emb, bin_emb], dim=1)
        
        x = self.transformer(x)
        
        # Pool and project
        x = x.mean(dim=1)
        x = self.out_proj(x)
        
        return F.normalize(x, p=2, dim=1)

class MatchingModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        # business tower
        self.business_tower = TowerBlock(
            num_features=len(encoder.num_bus) + len(encoder.states),  # Include shipping regions
            vocab_dict=encoder.vocab_bus,
            bin_features=len(encoder.bin_bus)
        )
        
        self.threepl_tower = TowerBlock(
            num_features=len(encoder.num_tpl) + len(encoder.states),  # Include covered states
            vocab_dict=encoder.vocab_tpl,
            bin_features=len(encoder.bin_tpl)
        )

    def forward(self, business_data, threepl_data):
        bus_num, bus_cat, bus_bin = business_data
        tpl_num, tpl_cat, tpl_bin = threepl_data
        return self.business_tower(bus_num, bus_cat, bus_bin), self.threepl_tower(tpl_num, tpl_cat, tpl_bin)