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

# Load model checkpoint
checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'), weights_only=False)

# Extract model and encoder
model_state = checkpoint["model_state_dict"]
encoder = checkpoint["encoder_state"]

# Rebuild the model (must match architecture)
from _3pl_matching_model import MatchingModel

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
        
        # Add binary columns for businesses
        for prefix in binary_prefixes:
            for col in encoder.bin_bus:
                if col.startswith(prefix) and col not in bus_df.columns:
                    logger.warning(f"Missing business feature: {col}, filling with 0")
                    bus_df[col] = 0
        
        # Add binary columns for 3PLs
        for prefix in binary_prefixes:
            for col in encoder.bin_tpl:
                if col.startswith(prefix) and col not in tpl_df.columns:
                    logger.warning(f"Missing 3PL feature: {col}, filling with 0")
                    tpl_df[col] = 0
        
        # Encode features
        logger.info("Encoding business data")
        bus_num, bus_cat, bus_bin = encoder.encode_business(bus_df)
        
        logger.info("Encoding 3PL data")
        tpl_num, tpl_cat, tpl_bin = encoder.encode_3pl(tpl_df)
        
        # Reshape tensors
        bus_data = (bus_num, bus_cat, bus_bin)
        tpl_data = (tpl_num, tpl_cat, tpl_bin)
        
        logger.info("Generating prediction")
        with torch.no_grad():
            bus_emb, tpl_emb = model(bus_data, tpl_data)
            similarity = (torch.cosine_similarity(bus_emb, tpl_emb) + 1) / 2
            match_score = similarity.item()
        
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
