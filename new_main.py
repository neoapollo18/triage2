from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np
import logging
import json
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Initialize app
app = FastAPI(title="3PL Matching API", description="API for matching businesses with 3PL providers")

# Import the model classes
from _3pl_matching_model import FeatureEncoder, MatchingModel

# Create a reconstructable encoder class - this is crucial for deployment
class EncoderState:
    """A deployment-friendly version of FeatureEncoder that works with pickle across modules"""
    
    def __init__(self, num_bus, cat_bus, bin_bus, num_tpl, cat_tpl, bin_tpl, 
                 states, state_to_idx, vocab_bus, vocab_tpl):
        self.num_bus = num_bus
        self.cat_bus = cat_bus
        self.bin_bus = bin_bus
        self.num_tpl = num_tpl
        self.cat_tpl = cat_tpl 
        self.bin_tpl = bin_tpl
        self.states = states
        self.state_to_idx = state_to_idx
        self.vocab_bus = vocab_bus
        self.vocab_tpl = vocab_tpl
    
    def encode_business(self, data):
        """Encode business features to model-ready format"""
        # Get numeric features
        num_features = torch.tensor(data[self.num_bus].fillna(0).values, dtype=torch.float32)
        
        # Get categorical features
        cat_features = []
        for col in self.cat_bus:
            if col in data.columns:
                # Get unique vocab index for each category
                idx = torch.tensor([
                    self.vocab_bus.get(col, {}).get(str(val), 0) 
                    for val in data[col].fillna('unknown').values
                ], dtype=torch.long)
                cat_features.append(idx)
            else:
                # Use zeros if column missing
                cat_features.append(torch.zeros(len(data), dtype=torch.long))
        
        # Get binary features
        bin_features = torch.tensor(data[self.bin_bus].fillna(0).values, dtype=torch.float32)
        
        # Process shipping regions if present
        if 'top_shipping_regions' in data.columns:
            region_weights = torch.zeros(len(data), len(self.states), dtype=torch.float32)
            for i, regions_str in enumerate(data['top_shipping_regions']):
                try:
                    # Parse region weights format "STATE:WEIGHT;STATE:WEIGHT;..."
                    if pd.notna(regions_str):
                        pairs = regions_str.split(';')
                        for pair in pairs:
                            if ':' in pair:
                                state, weight = pair.split(':')
                                if state in self.state_to_idx:
                                    state_idx = self.state_to_idx[state]
                                    region_weights[i, state_idx] = float(weight)
                except:
                    # Default handling for parse errors
                    pass
            
            # Combine numeric features with region weights
            num_features = torch.cat([num_features, region_weights], dim=1)
        
        return num_features, cat_features, bin_features
    
    def encode_3pl(self, data):
        """Encode 3PL features to model-ready format"""
        # Get numeric features
        num_features = torch.tensor(data[self.num_tpl].fillna(0).values, dtype=torch.float32)
        
        # Get categorical features
        cat_features = []
        for col in self.cat_tpl:
            if col in data.columns:
                # Get unique vocab index for each category
                idx = torch.tensor([
                    self.vocab_tpl.get(col, {}).get(str(val), 0) 
                    for val in data[col].fillna('unknown').values
                ], dtype=torch.long)
                cat_features.append(idx)
            else:
                # Use zeros if column missing
                cat_features.append(torch.zeros(len(data), dtype=torch.long))
        
        # Get binary features
        bin_features = torch.tensor(data[self.bin_tpl].fillna(0).values, dtype=torch.float32)
        
        # Process covered states if present
        if 'covered_states' in data.columns:
            state_coverage = torch.zeros(len(data), len(self.states), dtype=torch.float32)
            for i, states_str in enumerate(data['covered_states']):
                try:
                    # Parse covered states format "STATE;STATE;STATE;..."
                    if pd.notna(states_str):
                        states = states_str.split(';')
                        for state in states:
                            if state in self.state_to_idx:
                                state_idx = self.state_to_idx[state]
                                state_coverage[i, state_idx] = 1.0
                except:
                    # Default handling for parse errors
                    pass
            
            # Combine numeric features with state coverage
            num_features = torch.cat([num_features, state_coverage], dim=1)
        
        return num_features, cat_features, bin_features

# Load model components
try:
    logger.info("Loading model checkpoint...")
    
    # Load the original model first to extract components
    # This step would be skipped in production - the components would be pre-extracted
    try:
        # Try loading with PyTorch 2.6+ approach
        original_checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'), weights_only=False)
    except TypeError:
        # Fall back to older PyTorch versions
        original_checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))
    
    # Extract components
    model_state = original_checkpoint["model_state_dict"]
    encoder_orig = original_checkpoint["encoder_state"]
    
    # Convert the encoder to our deployment-friendly version
    # In production, this would be pre-processed and loaded from files
    encoder = EncoderState(
        num_bus=encoder_orig.num_bus,
        cat_bus=encoder_orig.cat_bus,
        bin_bus=encoder_orig.bin_bus,
        num_tpl=encoder_orig.num_tpl,
        cat_tpl=encoder_orig.cat_tpl,
        bin_tpl=encoder_orig.bin_tpl,
        states=encoder_orig.states,
        state_to_idx=encoder_orig.state_to_idx,
        vocab_bus=encoder_orig.vocab_bus,
        vocab_tpl=encoder_orig.vocab_tpl
    )
    
    # Build the model
    model = MatchingModel(encoder)
    model.load_state_dict(model_state)
    model.eval()
    
    logger.info(f"Model loaded successfully! Was trained for {original_checkpoint.get('epoch')} epochs")
    logger.info(f"Best validation AUC: {original_checkpoint.get('best_auc')}")
    
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
    raise

# Define request/response formats
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
