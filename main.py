from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(title="3PL Matching API", description="API for matching businesses with 3PL providers")

# Import the model classes first
from _3pl_matching_model import FeatureEncoder, MatchingModel

# Add current directory to path to help with module imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the model classes first - this is critical
from _3pl_matching_model import FeatureEncoder, MatchingModel

# Import the safe encoder functions
from safe_encoder import safe_encode_business, safe_encode_3pl

# Critical fix: Register the FeatureEncoder class in the module namespace
# This allows pickle to find the class during unpickling
import __main__
__main__.FeatureEncoder = FeatureEncoder

# Helper function to handle unknown categorical values
def safe_encode_categorical(encoder, df_col, categorical_name):
    """Safely encode categorical values to ensure they are within valid embedding range"""
    # Get the vocabulary size for this categorical feature
    vocab_size = encoder.vocab_bus.get(categorical_name) if categorical_name in encoder.vocab_bus else encoder.vocab_tpl.get(categorical_name)
    
    if vocab_size is None:
        logger.error(f"Cannot find vocabulary size for {categorical_name}")
        return None
    
    # We need to make sure the encoded values are in range [0, vocab_size-1]
    # This is needed for the embedding layer which expects indices in this range
    max_idx = vocab_size - 1
    
    # For an ordinal encoder, default to 0 (typically the most common value)
    # This avoids index out of range errors in the embedding layer
    return min(max_idx, df_col)

# Load model checkpoint with appropriate error handling
try:
    logger.info("Attempting to load model checkpoint")
    
    # Now that we've registered FeatureEncoder in __main__, the load should work
    try:
        # Try PyTorch 2.6+ approach with weights_only=False
        checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'), weights_only=False)
        logger.info("Model loaded successfully with weights_only=False parameter")
    except TypeError:
        # Fall back to older PyTorch versions without weights_only parameter
        checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))
        logger.info("Model loaded successfully with basic method")
    
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
    raise

# Extract model and encoder
model_state = checkpoint["model_state_dict"]
encoder = checkpoint["encoder_state"]

# Rebuild the model (must match architecture)

model = MatchingModel(encoder)
model.load_state_dict(model_state)
model.eval()

# --------------------------------
# Define request/response formats
# --------------------------------

class BusinessInput(BaseModel):
    business: dict
    threepl: dict

# Add health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict_match(data: BusinessInput):
    try:
        # Parse business and 3PL input dictionaries
        business_data = data.business
        threepl_data = data.threepl
        
        logger.info(f"Received business data: {business_data}")
        logger.info(f"Received 3PL data: {threepl_data}")
        
        # Convert to DataFrames
        bus_df = pd.DataFrame([business_data])
        tpl_df = pd.DataFrame([threepl_data])
        
        # Define expected features
        # Business numeric features
        num_bus = [
            'sku_count', 'shipping_volume', 'order_speed_expectation',
            'log_avg_order_value', 'daily_order_variance', 'return_rate_pct',
            'avg_sku_turnover_days', 'avg_package_weight_kg', 'year_founded',
            'business_age_yrs', 'growth_velocity_pct'
        ]
        
        # Business categorical features
        cat_bus = [
            'business_type', 'target_market', 'temperature_control_needed',
            'dimensional_weight_class'
        ]
        
        # 3PL numeric features
        num_tpl = [
            'min_monthly_volume', 'max_monthly_volume', 'average_shipping_time_days',
            'dock_to_stock_hours', 'max_daily_orders', 'picking_accuracy_pct',
            'available_storage_sqft', 'num_warehouses'
        ]
        
        # 3PL categorical features
        cat_tpl = ['headquarters_state', 'service_coverage']
        
        # Fill missing numeric values with 0
        for col in num_bus:
            if col not in bus_df.columns:
                logger.warning(f"Missing business feature: {col}, filling with 0")
                bus_df[col] = 0
                
        for col in num_tpl:
            if col not in tpl_df.columns:
                logger.warning(f"Missing 3PL feature: {col}, filling with 0")
                tpl_df[col] = 0
                
        # Fill missing categorical values with 'unknown'
        for col in cat_bus:
            if col not in bus_df.columns:
                logger.warning(f"Missing business feature: {col}, filling with 'unknown'")
                bus_df[col] = 'unknown'
                
        for col in cat_tpl:
            if col not in tpl_df.columns:
                logger.warning(f"Missing 3PL feature: {col}, filling with 'unknown'")
                tpl_df[col] = 'unknown'
        
        # Set business_id and 3pl_id if not provided
        if 'business_id' not in bus_df.columns:
            bus_df['business_id'] = 'unknown'
            
        if '3pl_id' not in tpl_df.columns:
            tpl_df['3pl_id'] = 'unknown'
        
        # Fill missing binary features with 0
        # Business binary features
        binary_prefixes = ['industry_', 'growth_', 'tech_', 'service_', 'specialty_']
        
        # Add binary columns for businesses using EXACT column names from encoder
        for col in encoder.bin_bus:
            if col not in bus_df.columns:
                logger.warning(f"Missing business feature: {col}, filling with 0")
                bus_df[col] = 0
        
        # Add binary columns for 3PLs using EXACT column names from encoder
        for col in encoder.bin_tpl:
            if col not in tpl_df.columns:
                logger.warning(f"Missing 3PL feature: {col}, filling with 0")
                tpl_df[col] = 0
        
        # Handle missing shipping regions and covered states
        if 'top_shipping_regions' not in bus_df.columns:
            logger.warning("Missing top_shipping_regions, using default NY:0.98;NC:0.02")
            bus_df['top_shipping_regions'] = "NY:0.98;NC:0.02"
        
        if 'covered_states' not in tpl_df.columns:
            # Get states from service_coverage if available
            if 'service_coverage' in tpl_df.columns:
                logger.warning("Missing covered_states, using service_coverage value")
                tpl_df['covered_states'] = tpl_df['service_coverage']
            else:
                logger.warning("Missing covered_states, using default NY;NC")
                tpl_df['covered_states'] = "NY;NC"
        
        # Ensure data types are correct before encoding
        logger.info("Preparing data for encoding")
        
        # Convert numeric columns to proper type
        for col in encoder.num_bus:
            bus_df[col] = pd.to_numeric(bus_df[col], errors='coerce').fillna(0)
            
        for col in encoder.num_tpl:
            tpl_df[col] = pd.to_numeric(tpl_df[col], errors='coerce').fillna(0)
        
        # Ensure binary columns are integers
        for col in encoder.bin_bus:
            bus_df[col] = pd.to_numeric(bus_df[col], errors='coerce').fillna(0).astype(int)
            
        for col in encoder.bin_tpl:
            tpl_df[col] = pd.to_numeric(tpl_df[col], errors='coerce').fillna(0).astype(int)
        
        # Process categorical data
        for col in encoder.cat_bus:
            bus_df[col] = bus_df[col].astype(str).fillna('unknown')
            
        for col in encoder.cat_tpl:
            tpl_df[col] = tpl_df[col].astype(str).fillna('unknown')
        
        try:
            # Use our safe encoder functions that guarantee valid tensor dimensions
            logger.info("Using safe encoder functions for API prediction")
            
            # Encode business data with guaranteed shapes [1,21], [1,4], [1,34]
            bus_num, bus_cat, bus_bin = safe_encode_business(encoder, bus_df)
            logger.info(f"Business tensor shapes: num={bus_num.shape}, cat={bus_cat.shape}, bin={bus_bin.shape}")
            
            # Encode 3PL data with guaranteed shapes [1,18], [1,2], [1,19]
            tpl_num, tpl_cat, tpl_bin = safe_encode_3pl(encoder, tpl_df)
            logger.info(f"3PL tensor shapes: num={tpl_num.shape}, cat={tpl_cat.shape}, bin={tpl_bin.shape}")
            
            # Reshape tensors
            bus_data = (bus_num, bus_cat, bus_bin)
            tpl_data = (tpl_num, tpl_cat, tpl_bin)
            
            logger.info("Data encoded successfully with guaranteed tensor shapes")
        except Exception as encoding_error:
            logger.error(f"Error during data encoding: {encoding_error}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error encoding data: {str(encoding_error)}")
        
        logger.info("Generating prediction")
        with torch.no_grad():
            match_score = model(bus_data, tpl_data).item()
        
        # Prepare result
        result = {
            "match_score": float(match_score),
            "business_id": business_data.get("business_id", "unknown"),
            "3pl_id": threepl_data.get("3pl_id", "unknown")
        }
        
        logger.info(f"Prediction result: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # Return error as HTTP 500 with details
        raise HTTPException(status_code=500, detail=str(e))
