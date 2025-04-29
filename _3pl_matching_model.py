# -------------------------------------------------------------
# 3PL ⇄ Business Matching – Optimized Two-Tower Architecture
# -------------------------------------------------------------
# Key features:
#   • Efficient embedding dimension (32) for categorical features
#   • Separate numeric and categorical processing
#   • Transformer-based feature interaction
#   • Contrastive learning with hard negatives
#   • Batch size 256 with 6000 labeled pairs
#   • Early stopping and learning rate scheduling
# -------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from datetime import datetime
import wandb
from category_encoders import OrdinalEncoder
from tqdm import tqdm

# ------------------------------
# Configuration
# ------------------------------

class Config:
    SEED = 42
    EMBEDDING_DIM = 32
    TRANSFORMER_LAYERS = 2
    ATTENTION_HEADS = 4
    DROPOUT = 0.1
    BATCH_SIZE = 256
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2
    EPOCHS = 25
    FOLDS = 5
    EARLY_STOPPING = 5
    WARMUP_EPOCHS = 5


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
seed_everything(Config.SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------
# Feature Processing
# ------------------------------

class FeatureEncoder:
    def __init__(self, bus_df: pd.DataFrame, tpl_df: pd.DataFrame):
        # business numeric features
        self.num_bus = [
            'sku_count', 'shipping_volume', 'order_speed_expectation',
            'log_avg_order_value', 'daily_order_variance', 'return_rate_pct',
            'avg_sku_turnover_days', 'avg_package_weight_kg', 'year_founded',
            'business_age_yrs', 'growth_velocity_pct'
        ]
        
        # nusiness categorical features
        self.cat_bus = [
            'business_type', 'target_market', 'temperature_control_needed',
            'dimensional_weight_class'
        ]
        

        self.bin_bus = [
            col for col in bus_df.columns 
            if col.startswith(('industry_', 'growth_', 'tech_', 'service_', 'specialty_'))
        ]
        

        self.num_tpl = [
            'min_monthly_volume', 'max_monthly_volume', 'average_shipping_time_days',
            'dock_to_stock_hours', 'max_daily_orders', 'picking_accuracy_pct',
            'available_storage_sqft', 'num_warehouses'
        ]
        
        self.cat_tpl = ['headquarters_state', 'service_coverage']
        

        self.bin_tpl = [
            col for col in tpl_df.columns 
            if col.startswith(('tech_', 'service_', 'specialty_'))
        ]
        
        self.states = sorted(list(set(
            state for _, row in bus_df.iterrows()
            for state_weight in row['top_shipping_regions'].split(';')
            for state in [state_weight.split(':')[0]]
        )))
        

        for col in self.bin_bus:
            bus_df[col] = pd.to_numeric(bus_df[col], errors='coerce').fillna(0).astype(int)
        for col in self.bin_tpl:
            tpl_df[col] = pd.to_numeric(tpl_df[col], errors='coerce').fillna(0).astype(int)
        

        self.scaler_bus = StandardScaler().fit(bus_df[self.num_bus])
        self.scaler_tpl = StandardScaler().fit(tpl_df[self.num_tpl])
        self.enc_bus = OrdinalEncoder().fit(bus_df[self.cat_bus])
        self.enc_tpl = OrdinalEncoder().fit(tpl_df[self.cat_tpl])
        

        self.vocab_bus = {c: int(bus_df[c].nunique()) + 1 for c in self.cat_bus}
        self.vocab_tpl = {c: int(tpl_df[c].nunique()) + 1 for c in self.cat_tpl}

    def _parse_shipping_regions(self, regions_str):
        """Convert shipping regions string to state weight vector"""
        weights = {state: 0.0 for state in self.states}
        if pd.isna(regions_str):
            return list(weights.values())
        
        for pair in regions_str.split(';'):
            if ':' in pair:
                state, weight = pair.split(':')
                weights[state] = float(weight)
        
        return list(weights.values())

    def _parse_covered_states(self, states_str):
        """Convert covered states string to binary vector"""
        covered = {state: 0 for state in self.states}
        if pd.isna(states_str):
            return list(covered.values())
        
        for state in states_str.split(';'):
            if state in covered:
                covered[state] = 1
        
        return list(covered.values())

    def encode_business(self, df):
        for col in self.bin_bus:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        num = torch.tensor(self.scaler_bus.transform(df[self.num_bus]), dtype=torch.float32)
        

        cat = torch.tensor(self.enc_bus.transform(df[self.cat_bus]).values, dtype=torch.long)

        bin = torch.tensor(df[self.bin_bus].values, dtype=torch.float32)
        

        shipping_weights = torch.tensor([self._parse_shipping_regions(row['top_shipping_regions']) 
                                      for _, row in df.iterrows()], dtype=torch.float32)
        
        # Combine numeric features with shipping weights
        num = torch.cat([num, shipping_weights], dim=1)
        
        return num, cat, bin

    def encode_3pl(self, df):

        for col in self.bin_tpl:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        num = torch.tensor(self.scaler_tpl.transform(df[self.num_tpl]), dtype=torch.float32)
        

        cat = torch.tensor(self.enc_tpl.transform(df[self.cat_tpl]).values, dtype=torch.long)
        

        bin = torch.tensor(df[self.bin_tpl].values, dtype=torch.float32)
        

        covered_states = torch.tensor([self._parse_covered_states(row['covered_states']) 
                                    for _, row in df.iterrows()], dtype=torch.float32)
        

        num = torch.cat([num, covered_states], dim=1)
        
        return num, cat, bin

# ------------------------------
# model architecture
# ------------------------------

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
        # -rocess numeric features
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
    def __init__(self, encoder: FeatureEncoder):
        super().__init__()
        self.business_tower = TowerBlock(
            num_features=len(encoder.num_bus) + len(encoder.states),
            vocab_dict=encoder.vocab_bus,
            bin_features=len(encoder.bin_bus)
        )
        self.threepl_tower = TowerBlock(
            num_features=len(encoder.num_tpl) + len(encoder.states),
            vocab_dict=encoder.vocab_tpl,
            bin_features=len(encoder.bin_tpl)
        )

        self.pred_head = nn.Sequential(
            nn.Linear(Config.EMBEDDING_DIM * 3, 128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, business_data, threepl_data):
        bus_num, bus_cat, bus_bin = business_data
        tpl_num, tpl_cat, tpl_bin = threepl_data

        bus_emb = self.business_tower(bus_num, bus_cat, bus_bin)
        tpl_emb = self.threepl_tower(tpl_num, tpl_cat, tpl_bin)

        # Combine embeddings
        combined = torch.cat([bus_emb, tpl_emb, bus_emb * tpl_emb], dim=1)
        score = self.pred_head(combined)

        return score.squeeze(-1)
# Dataset
# ------------------------------

class MatchingDataset(Dataset):
    def __init__(self, pairs, businesses_df, threepls_df, encoder, augment=False):
        self.pairs = pairs
        self.businesses_df = businesses_df
        self.threepls_df = threepls_df
        self.encoder = encoder
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def augment_features(self, num):
        if self.augment and torch.rand(1).item() < 0.3:
            return num + torch.randn_like(num) * 0.05
        return num

    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]
        
        # Get business data
        business = self.businesses_df[self.businesses_df.business_id == pair.business_id].iloc[0]
        bus_num, bus_cat, bus_bin = self.encoder.encode_business(business.to_frame().T)
        
        # Get 3PL data
        threepl = self.threepls_df[self.threepls_df['3pl_id'] == pair['3pl_id']].iloc[0]
        tpl_num, tpl_cat, tpl_bin = self.encoder.encode_3pl(threepl.to_frame().T)
        
        # Apply augmentation to numeric features
        if self.augment:
            bus_num = self.augment_features(bus_num)
            tpl_num = self.augment_features(tpl_num)
        
        return (
            bus_num.squeeze(0), bus_cat.squeeze(0), bus_bin.squeeze(0),
            tpl_num.squeeze(0), tpl_cat.squeeze(0), tpl_bin.squeeze(0),
            torch.tensor(pair.composite_score, dtype=torch.float32)
        )

def collate_fn(batch):
    bus_num, bus_cat, bus_bin, tpl_num, tpl_cat, tpl_bin, labels = zip(*batch)
    return (
        (torch.stack(bus_num), torch.stack(bus_cat), torch.stack(bus_bin)),
        (torch.stack(tpl_num), torch.stack(tpl_cat), torch.stack(tpl_bin)),
        torch.stack(labels)
    )

# ------------------------------
# Training Loop
# ------------------------------

def train_cv(businesses_df, threepls_df, labeled_pairs, config=Config):
    encoder = FeatureEncoder(businesses_df, threepls_df)
    
    # Convert labels to binary for stratification
    labels_binary = (labeled_pairs.composite_score >= 0.5).astype(int)
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=config.FOLDS, shuffle=True, random_state=config.SEED)
    
    # Initialize wandb with config dictionary
    wandb_config = {
        'seed': config.SEED,
        'embedding_dim': config.EMBEDDING_DIM,
        'transformer_layers': config.TRANSFORMER_LAYERS,
        'attention_heads': config.ATTENTION_HEADS,
        'dropout': config.DROPOUT,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'weight_decay': config.WEIGHT_DECAY,
        'epochs': config.EPOCHS,
        'folds': config.FOLDS,
        'early_stopping': config.EARLY_STOPPING,
        'warmup_epochs': config.WARMUP_EPOCHS
    }
    
    wandb.init(
        project="3pl-matching",
        name=f"two_tower_{datetime.now():%Y%m%d_%H%M}",
        config=wandb_config
    )
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(labeled_pairs, labels_binary), 1):
        print(f"\nFold {fold}/{config.FOLDS}")
        
        # Create datasets
        train_pairs = labeled_pairs.iloc[train_idx]
        val_pairs = labeled_pairs.iloc[val_idx]
        
        train_dataset = MatchingDataset(train_pairs, businesses_df, threepls_df, encoder, augment=True)
        val_dataset = MatchingDataset(val_pairs, businesses_df, threepls_df, encoder, augment=False)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        # Initialize model and training components
        model = MatchingModel(encoder).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.LEARNING_RATE,
            epochs=config.EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
        
        # Training loop
        best_auc = 0
        patience = 0
        
        for epoch in range(config.EPOCHS):
            # Training phase
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Train]')
            for i, (bus_data, tpl_data, labels) in enumerate(train_pbar):
                # Move data to device
                bus_data = tuple(t.to(device) for t in bus_data)
                tpl_data = tuple(t.to(device) for t in tpl_data)
                labels = labels.to(device)
                
                # Forward pass
                predicted_score = model(bus_data, tpl_data)
                loss = F.binary_cross_entropy(predicted_score, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Track predictions
                train_preds.extend(predicted_score.detach().cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            train_loss /= len(train_loader)
            train_auc = roc_auc_score((np.array(train_labels) >= 0.5).astype(int), train_preds)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Val]')
            with torch.no_grad():
                for bus_data, tpl_data, labels in val_pbar:
                    bus_data = tuple(t.to(device) for t in bus_data)
                    tpl_data = tuple(t.to(device) for t in tpl_data)
                    labels = labels.to(device)
                    
                    predicted_score = model(bus_data, tpl_data)
                    loss = F.binary_cross_entropy(predicted_score, labels)
                    val_loss += loss.item()
                    
                    # Track predictions
                    val_preds.extend(predicted_score.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            val_loss /= len(val_loader)
            val_auc = roc_auc_score((np.array(val_labels) >= 0.5).astype(int), val_preds)
            
            # Log metrics
            wandb.log({
                'fold': fold,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_auc': train_auc,
                'val_loss': val_loss,
                'val_auc': val_auc,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
            
            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'encoder_state': encoder,
                    'best_auc': best_auc
                }, f'best_model_fold{fold}.pt')
                patience = 0
            else:
                patience += 1
                if patience >= config.EARLY_STOPPING:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        fold_scores.append(best_auc)
        print(f"Fold {fold} Best AUC: {best_auc:.4f}")
    
    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    print(f"\nCross-validation results:")
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    wandb.log({
        'cv_mean_auc': mean_auc,
        'cv_std_auc': std_auc
    })
    wandb.finish()

def run_once(businesses_df, threepls_df, labeled_pairs, rep):
    """Run a single training iteration without cross-validation"""
    # Use the Config class directly
    config = Config
    
    encoder = FeatureEncoder(businesses_df, threepls_df)
    
    # Convert labels to binary for stratification
    labels_binary = (labeled_pairs.composite_score >= 0.5).astype(int)
    
    # Split data into train and validation
    train_idx, val_idx = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED).split(labeled_pairs, labels_binary))
    train_pairs = labeled_pairs.iloc[train_idx]
    val_pairs = labeled_pairs.iloc[val_idx]
    
    # Initialize wandb with config dictionary
    wandb_config = {
        'seed': config.SEED,
        'embedding_dim': config.EMBEDDING_DIM,
        'transformer_layers': config.TRANSFORMER_LAYERS,
        'attention_heads': config.ATTENTION_HEADS,
        'dropout': config.DROPOUT,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'weight_decay': config.WEIGHT_DECAY,
        'epochs': config.EPOCHS,
        'folds': 1,  # Single fold for run_once
        'early_stopping': config.EARLY_STOPPING,
        'warmup_epochs': config.WARMUP_EPOCHS,
        'repetition': rep
    }
    
    wandb.init(
        project="3pl-matching",
        name=f"rep_{rep}_{datetime.now():%Y%m%d_%H%M}",
        config=wandb_config
    )
    
    # Create datasets
    train_dataset = MatchingDataset(train_pairs, businesses_df, threepls_df, encoder, augment=True)
    val_dataset = MatchingDataset(val_pairs, businesses_df, threepls_df, encoder, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Initialize model and training components
    model = MatchingModel(encoder).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    # loop
    best_auc = 0
    patience = 0
    best_metrics = None
    all_val_aucs = []  # track all validation AUCs
    
    for epoch in range(config.EPOCHS):
        # training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Train]')
        for i, (bus_data, tpl_data, labels) in enumerate(train_pbar):
            # Move data to device
            bus_data = tuple(t.to(device) for t in bus_data)
            tpl_data = tuple(t.to(device) for t in tpl_data)
            labels = labels.to(device)
            
            # Forward pass
            predicted_score = model(bus_data, tpl_data)
            loss = F.binary_cross_entropy(predicted_score, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track predictions
            train_preds.extend(predicted_score.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_auc = roc_auc_score((np.array(train_labels) >= 0.5).astype(int), train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Val]')
        with torch.no_grad():
            for bus_data, tpl_data, labels in val_pbar:
                bus_data = tuple(t.to(device) for t in bus_data)
                tpl_data = tuple(t.to(device) for t in tpl_data)
                labels = labels.to(device)
                
                predicted_score = model(bus_data, tpl_data)
                loss = F.binary_cross_entropy(predicted_score, labels)
                val_loss += loss.item()
                
                # Track predictions
                val_preds.extend(predicted_score.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        val_auc = roc_auc_score((np.array(val_labels) >= 0.5).astype(int), val_preds)
        all_val_aucs.append(val_auc)  # Track all validation AUCs
        
        # Log metrics
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_auc': train_auc,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        # Save best model and metrics
        if val_auc > best_auc:
            best_auc = val_auc
            best_metrics = {
                'train_loss': train_loss,
                'train_auc': train_auc,
                'val_loss': val_loss,
                'val_auc': val_auc,
                'epoch': epoch + 1
            }
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state': encoder,
                'best_auc': best_auc
            }, 'best_model.pt')
            patience = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    wandb.finish()
    

    mean_auc = np.mean(all_val_aucs)
    std_auc = np.std(all_val_aucs)
    
    return mean_auc, std_auc  # return both mean and std of validation AUCs

if __name__ == "__main__":
    print("Loading datasets...")
    try:
        businesses_df = pd.read_csv('businesses.csv')
        threepls_df = pd.read_csv('3pls.csv')
        labeled_pairs = pd.read_csv('labeled_data.csv')
        
        print("\nDataset shapes:")
        print(f"Businesses: {businesses_df.shape}")
        print(f"3PLs: {threepls_df.shape}")
        print(f"Labeled pairs: {labeled_pairs.shape}")
        
        print("\nTraining model on real data...")
        mean_auc, std_auc = run_once(businesses_df, threepls_df, labeled_pairs, rep=1)
        print(f"\nFinal Validation AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required data file - {e}")
        print("Please ensure the following files exist in the current directory:")
        print("- businesses.csv")
        print("- 3pls.csv")
        print("- labeled_data.csv")
    except Exception as e:
        print(f"Error during training: {e}")
