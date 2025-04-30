import torch
import pandas as pd
import numpy as np
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path to help with module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import model classes
from _3pl_matching_model import FeatureEncoder, MatchingModel

# Register FeatureEncoder class in the __main__ module namespace
import __main__
__main__.FeatureEncoder = FeatureEncoder

def debug_model_embeddings():
    """Debug the model embeddings to understand the shape and size issues"""
    try:
        # Step 1: Load the model
        logger.info("Loading model checkpoint")
        checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))
        encoder = checkpoint["encoder_state"]
        model = MatchingModel(encoder)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info(f"Model loaded: epoch={checkpoint.get('epoch')}, auc={checkpoint.get('best_auc'):.4f}")
        
        # Step 2: Print detailed information about embedding sizes
        logger.info("\n===== EMBEDDING LAYER DETAILS =====")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                logger.info(f"Embedding: {name}, num_embeddings={module.num_embeddings}, embedding_dim={module.embedding_dim}")
        
        # Step 3: Print vocabulary sizes from encoder
        logger.info("\n===== VOCABULARY SIZES =====")
        logger.info(f"Business categorical features: {encoder.cat_bus}")
        for col in encoder.cat_bus:
            logger.info(f"{col}: vocab_size={encoder.vocab_bus[col]}")
            
        logger.info(f"3PL categorical features: {encoder.cat_tpl}")
        for col in encoder.cat_tpl:
            logger.info(f"{col}: vocab_size={encoder.vocab_tpl[col]}")
        
        # Step 4: Create minimal valid input data
        logger.info("\n===== CREATING MINIMAL VALID TEST DATA =====")
        
        # Create minimal business data with known values in range
        bus_data = {}
        
        # Numeric features
        for col in encoder.num_bus:
            bus_data[col] = [0.0]  # Default to 0
            
        # Categorical features - use first category for each (index 0)
        for col in encoder.cat_bus:
            bus_data[col] = ["unknown"]
            
        # Binary features
        for col in encoder.bin_bus:
            bus_data[col] = [0]
            
        # Add shipping regions
        bus_data["top_shipping_regions"] = ["NY:1.0"]
        
        # Create minimal 3PL data
        tpl_data = {}
        
        # Numeric features
        for col in encoder.num_tpl:
            tpl_data[col] = [0.0]
            
        # Categorical features - use first category for each (index 0)
        for col in encoder.cat_tpl:
            tpl_data[col] = ["unknown"]
            
        # Binary features
        for col in encoder.bin_tpl:
            tpl_data[col] = [0]
            
        # Add covered states
        tpl_data["covered_states"] = ["NY;CA"]
        
        # Convert to DataFrames
        bus_df = pd.DataFrame(bus_data)
        tpl_df = pd.DataFrame(tpl_data)
        
        logger.info("Test data created with all zeros and default values")
        
        # Step 5: Carefully encode with debugging information
        logger.info("\n===== ENCODING TEST DATA =====")
        
        # Business numeric features
        bus_num_values = encoder.scaler_bus.transform(bus_df[encoder.num_bus])
        logger.info(f"Business numeric shape after scaling: {bus_num_values.shape}")
        
        # Add shipping weights
        shipping_weights = np.array([encoder._parse_shipping_regions(row['top_shipping_regions']) 
                                    for _, row in bus_df.iterrows()])
        logger.info(f"Shipping weights shape: {shipping_weights.shape}")
        
        bus_num = torch.tensor(np.hstack([bus_num_values, shipping_weights]), dtype=torch.float32)
        logger.info(f"Final business numeric tensor shape: {bus_num.shape}")
        
        # Business binary features
        bus_bin = torch.tensor(bus_df[encoder.bin_bus].values, dtype=torch.float32)
        logger.info(f"Business binary tensor shape: {bus_bin.shape}")
        
        # Business categorical features
        cat_values = encoder.enc_bus.transform(bus_df[encoder.cat_bus])
        if hasattr(cat_values, 'values'):
            cat_values = cat_values.values
        cat_values = cat_values.astype(int)
        logger.info(f"Raw categorical values: {cat_values}")
        logger.info(f"Categorical shape: {cat_values.shape}")
        
        # Set all categorical indices to 0 to ensure they're in range
        cat_values[:] = 0
        
        bus_cat = torch.tensor(cat_values, dtype=torch.long)
        logger.info(f"Business categorical tensor shape: {bus_cat.shape}")
        
        # 3PL features
        tpl_num_values = encoder.scaler_tpl.transform(tpl_df[encoder.num_tpl])
        
        covered_states = np.array([encoder._parse_covered_states(row['covered_states']) 
                                    for _, row in tpl_df.iterrows()])
                                    
        tpl_num = torch.tensor(np.hstack([tpl_num_values, covered_states]), dtype=torch.float32)
        logger.info(f"3PL numeric tensor shape: {tpl_num.shape}")
        
        tpl_bin = torch.tensor(tpl_df[encoder.bin_tpl].values, dtype=torch.float32)
        logger.info(f"3PL binary tensor shape: {tpl_bin.shape}")
        
        cat_values = encoder.enc_tpl.transform(tpl_df[encoder.cat_tpl])
        if hasattr(cat_values, 'values'):
            cat_values = cat_values.values
        cat_values = cat_values.astype(int)
        
        # Set all categorical indices to 0 to ensure they're in range
        cat_values[:] = 0
        
        tpl_cat = torch.tensor(cat_values, dtype=torch.long)
        logger.info(f"3PL categorical tensor shape: {tpl_cat.shape}")
        
        # Step 6: Run model with forced valid data
        logger.info("\n===== RUNNING MODEL INFERENCE =====")
        with torch.no_grad():
            try:
                bus_data = (bus_num, bus_cat, bus_bin)
                tpl_data = (tpl_num, tpl_cat, tpl_bin)
                
                # Try to run business tower
                logger.info("Testing business tower...")
                bus_emb = model.business_tower(bus_num, bus_cat, bus_bin)
                logger.info(f"Business tower output shape: {bus_emb.shape}")
                
                # Try to run 3PL tower
                logger.info("Testing 3PL tower...")
                tpl_emb = model.threepl_tower(tpl_num, tpl_cat, tpl_bin)
                logger.info(f"3PL tower output shape: {tpl_emb.shape}")
                
                # Try full model inference
                logger.info("Testing full model inference...")
                match_score = model(bus_data, tpl_data).item()
                logger.info(f"Model prediction: {match_score:.4f}")
                
                print("\n✅ SUCCESS: Model inference works with minimal valid data!")
                print("Debug information shows the exact tensor shapes needed for inference")
                
                # Save the tensor shapes for reference in main.py
                print("\n===== TENSOR SHAPE REFERENCE =====")
                print(f"Business numeric: {bus_num.shape}")
                print(f"Business categorical: {bus_cat.shape}")
                print(f"Business binary: {bus_bin.shape}")
                print(f"3PL numeric: {tpl_num.shape}")
                print(f"3PL categorical: {tpl_cat.shape}")
                print(f"3PL binary: {tpl_bin.shape}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error during model inference: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
                
    except Exception as e:
        logger.error(f"Error in embedding debugging: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = debug_model_embeddings()
    sys.exit(0 if success else 1)
