"""
Safe encoder module for 3PL model deployment.
This provides robust encoding functions that ensure tensors match exactly 
what the model requires, avoiding index out of range errors.
"""
import numpy as np
import pandas as pd
import torch
import logging

logger = logging.getLogger(__name__)

def safe_encode_business(encoder, bus_df):
    """
    Safely encode business data for model inference with guaranteed valid dimensions
    Returns tensors with shapes:
        - num: [batch_size, 21]
        - cat: [batch_size, 4]
        - bin: [batch_size, 34]
    """
    # Ensure required columns exist
    for col in encoder.num_bus:
        if col not in bus_df.columns:
            bus_df[col] = 0
        bus_df[col] = pd.to_numeric(bus_df[col], errors='coerce').fillna(0)
        
    for col in encoder.cat_bus:
        if col not in bus_df.columns:
            bus_df[col] = "unknown"
        bus_df[col] = bus_df[col].astype(str).fillna('unknown')
        
    for col in encoder.bin_bus:
        if col not in bus_df.columns:
            bus_df[col] = 0
        bus_df[col] = pd.to_numeric(bus_df[col], errors='coerce').fillna(0).astype(int)
    
    if 'top_shipping_regions' not in bus_df.columns:
        bus_df['top_shipping_regions'] = "NY:1.0"
    
    # STEP 1: Numeric features with shipping regions
    try:
        # Transform basic numeric features
        bus_num_values = encoder.scaler_bus.transform(bus_df[encoder.num_bus])
        
        # Add shipping weights
        shipping_weights = np.array([
            encoder._parse_shipping_regions(row['top_shipping_regions']) 
            for _, row in bus_df.iterrows()
        ])
        
        # Combine and ensure correct shape
        bus_num = np.hstack([bus_num_values, shipping_weights])
        
        # Double-check dimensions and fix if necessary
        if bus_num.shape[1] < 21:
            # If shipping weights are missing states, pad with zeros
            padding = np.zeros((bus_num.shape[0], 21 - bus_num.shape[1]))
            bus_num = np.hstack([bus_num, padding])
        elif bus_num.shape[1] > 21:
            # If too many features, truncate
            bus_num = bus_num[:, :21]
            
        bus_num = torch.tensor(bus_num, dtype=torch.float32)
    except Exception as e:
        logger.error(f"Error encoding business numeric features: {e}")
        # Fallback to zeros with correct shape
        bus_num = torch.zeros((bus_df.shape[0], 21), dtype=torch.float32)
    
    # STEP 2: Categorical features
    try:
        # Transform categorical features 
        cat_values = encoder.enc_bus.transform(bus_df[encoder.cat_bus])
        if hasattr(cat_values, 'values'):
            cat_values = cat_values.values
        cat_values = cat_values.astype(int)
        
        # Ensure values are in range for each categorical feature
        for i, col in enumerate(encoder.cat_bus):
            vocab_size = encoder.vocab_bus[col]
            # Ensure index is within embedding range to avoid index error
            cat_values[:, i] = np.clip(cat_values[:, i], 0, vocab_size - 1)
            
        bus_cat = torch.tensor(cat_values, dtype=torch.long)
    except Exception as e:
        logger.error(f"Error encoding business categorical features: {e}")
        # Fallback to zeros (first category) with correct shape
        bus_cat = torch.zeros((bus_df.shape[0], 4), dtype=torch.long)
    
    # STEP 3: Binary features
    try:
        # Get binary features as tensor
        bin_values = bus_df[encoder.bin_bus].values
        
        # Ensure correct shape
        if bin_values.shape[1] < 34:
            # If missing binary features, pad with zeros
            padding = np.zeros((bin_values.shape[0], 34 - bin_values.shape[1]))
            bin_values = np.hstack([bin_values, padding])
        elif bin_values.shape[1] > 34:
            # If too many binary features, truncate
            bin_values = bin_values[:, :34]
            
        bus_bin = torch.tensor(bin_values, dtype=torch.float32)
    except Exception as e:
        logger.error(f"Error encoding business binary features: {e}")
        # Fallback to zeros with correct shape
        bus_bin = torch.zeros((bus_df.shape[0], 34), dtype=torch.float32)
    
    return bus_num, bus_cat, bus_bin


def safe_encode_3pl(encoder, tpl_df):
    """
    Safely encode 3PL data for model inference with guaranteed valid dimensions
    Returns tensors with shapes:
        - num: [batch_size, 18]
        - cat: [batch_size, 2]
        - bin: [batch_size, 19]
    """
    # Ensure required columns exist
    for col in encoder.num_tpl:
        if col not in tpl_df.columns:
            tpl_df[col] = 0
        tpl_df[col] = pd.to_numeric(tpl_df[col], errors='coerce').fillna(0)
        
    for col in encoder.cat_tpl:
        if col not in tpl_df.columns:
            tpl_df[col] = "unknown"
        tpl_df[col] = tpl_df[col].astype(str).fillna('unknown')
        
    for col in encoder.bin_tpl:
        if col not in tpl_df.columns:
            tpl_df[col] = 0
        tpl_df[col] = pd.to_numeric(tpl_df[col], errors='coerce').fillna(0).astype(int)
    
    if 'covered_states' not in tpl_df.columns:
        tpl_df['covered_states'] = "NY;CA"
    
    # STEP 1: Numeric features with covered states
    try:
        # Transform basic numeric features
        tpl_num_values = encoder.scaler_tpl.transform(tpl_df[encoder.num_tpl])
        
        # Add covered states
        covered_states = np.array([
            encoder._parse_covered_states(row['covered_states']) 
            for _, row in tpl_df.iterrows()
        ])
        
        # Combine and ensure correct shape
        tpl_num = np.hstack([tpl_num_values, covered_states])
        
        # Double-check dimensions and fix if necessary
        if tpl_num.shape[1] < 18:
            # If covered states are missing, pad with zeros
            padding = np.zeros((tpl_num.shape[0], 18 - tpl_num.shape[1]))
            tpl_num = np.hstack([tpl_num, padding])
        elif tpl_num.shape[1] > 18:
            # If too many features, truncate
            tpl_num = tpl_num[:, :18]
            
        tpl_num = torch.tensor(tpl_num, dtype=torch.float32)
    except Exception as e:
        logger.error(f"Error encoding 3PL numeric features: {e}")
        # Fallback to zeros with correct shape
        tpl_num = torch.zeros((tpl_df.shape[0], 18), dtype=torch.float32)
    
    # STEP 2: Categorical features
    try:
        # Transform categorical features 
        cat_values = encoder.enc_tpl.transform(tpl_df[encoder.cat_tpl])
        if hasattr(cat_values, 'values'):
            cat_values = cat_values.values
        cat_values = cat_values.astype(int)
        
        # Ensure values are in range for each categorical feature
        for i, col in enumerate(encoder.cat_tpl):
            vocab_size = encoder.vocab_tpl[col]
            # Ensure index is within embedding range to avoid index error
            cat_values[:, i] = np.clip(cat_values[:, i], 0, vocab_size - 1)
            
        tpl_cat = torch.tensor(cat_values, dtype=torch.long)
    except Exception as e:
        logger.error(f"Error encoding 3PL categorical features: {e}")
        # Fallback to zeros (first category) with correct shape
        tpl_cat = torch.zeros((tpl_df.shape[0], 2), dtype=torch.long)
    
    # STEP 3: Binary features
    try:
        # Get binary features as tensor
        bin_values = tpl_df[encoder.bin_tpl].values
        
        # Ensure correct shape
        if bin_values.shape[1] < 19:
            # If missing binary features, pad with zeros
            padding = np.zeros((bin_values.shape[0], 19 - bin_values.shape[1]))
            bin_values = np.hstack([bin_values, padding])
        elif bin_values.shape[1] > 19:
            # If too many binary features, truncate
            bin_values = bin_values[:, :19]
            
        tpl_bin = torch.tensor(bin_values, dtype=torch.float32)
    except Exception as e:
        logger.error(f"Error encoding 3PL binary features: {e}")
        # Fallback to zeros with correct shape
        tpl_bin = torch.zeros((tpl_df.shape[0], 19), dtype=torch.float32)
    
    return tpl_num, tpl_cat, tpl_bin
