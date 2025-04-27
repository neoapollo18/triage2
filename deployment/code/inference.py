import os
import json
import torch
import logging
import numpy as np
import pandas as pd
from io import StringIO

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Global variables to store model and processor
model = None
encoder = None

def model_fn(model_dir):
    """Load the PyTorch model and encoder from the model directory."""
    try:
        logger.info(f"Loading model from {model_dir}")
        
        # List directory contents for debugging
        logger.info(f"Directory contents: {os.listdir(model_dir)}")
        
        # Load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(model_dir, "best_model.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the saved model
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get the encoder
        encoder = checkpoint['encoder_state']
        
        # Extract model state dict
        model_state_dict = checkpoint['model_state_dict']
        
        # Import necessary classes
        from model_module import MatchingModel
        
        # Initialize model
        model = MatchingModel(encoder)
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return {'model': model, 'encoder': encoder, 'device': device}
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.exception("Stack trace:")
        raise

def input_fn(request_body, request_content_type):
    """Process the input data."""
    try:
        logger.info(f"Processing input with content type: {request_content_type}")
        
        if request_content_type == 'application/json':
            input_data = json.loads(request_body)
            logger.info(f"Received input: {json.dumps(input_data)[:200]}...")
            return input_data
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
    
    except Exception as e:
        logger.error(f"Error in input_fn: {str(e)}")
        logger.exception("Stack trace:")
        raise

def predict_fn(input_data, model_dict):
    """Generate a matching score between business and 3PL."""
    try:
        logger.info("Starting prediction")
        
        # Extract components from model_dict
        model = model_dict['model']
        encoder = model_dict['encoder']
        device = model_dict['device']
        
        # Extract business and 3PL data
        business_data = input_data.get('business', {})
        threepl_data = input_data.get('3pl', {})
        
        logger.info(f"Business data: {business_data}")
        logger.info(f"3PL data: {threepl_data}")
        
        # Convert to DataFrames
        business_df = pd.DataFrame([business_data])
        threepl_df = pd.DataFrame([threepl_data])
        
        # Ensure all required fields are present
        logger.info("Preparing data for model")
        
        # Set defaults for missing fields based on your feature encoder
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
            if col not in business_df:
                business_df[col] = 0
                
        for col in num_tpl:
            if col not in threepl_df:
                threepl_df[col] = 0
                
        # Fill missing categorical values with 'unknown'
        for col in cat_bus:
            if col not in business_df:
                business_df[col] = 'unknown'
                
        for col in cat_tpl:
            if col not in threepl_df:
                threepl_df[col] = 'unknown'
                
        # Encode the data using the provided encoder
        logger.info("Encoding business data")
        bus_num, bus_cat, bus_bin = encoder.encode_business(business_df)
        
        logger.info("Encoding 3PL data")
        tpl_num, tpl_cat, tpl_bin = encoder.encode_3pl(threepl_df)
        
        # Move to device
        bus_data = (
            bus_num.to(device),
            bus_cat.to(device),
            bus_bin.to(device)
        )
        
        tpl_data = (
            tpl_num.to(device),
            tpl_cat.to(device),
            tpl_bin.to(device)
        )
        
        # Get embeddings
        logger.info("Generating embeddings")
        with torch.no_grad():
            bus_emb, tpl_emb = model(bus_data, tpl_data)
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(bus_emb, tpl_emb)
            
            # Scale to [0, 1]
            score = (similarity + 1) / 2
            
        # Prepare result
        result = {
            'match_score': float(score.cpu().numpy()[0]),
            'business_id': business_data.get('business_id', ''),
            '3pl_id': threepl_data.get('3pl_id', '')
        }
        
        logger.info(f"Prediction result: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error in predict_fn: {str(e)}")
        logger.exception("Stack trace:")
        # Return error rather than raising to avoid endpoint failure
        return {
            'error': str(e),
            'success': False
        }

def output_fn(prediction, response_content_type):
    """Format the prediction output."""
    try:
        logger.info(f"Formatting output with content type: {response_content_type}")
        
        if response_content_type == 'application/json':
            return json.dumps(prediction)
        else:
            raise ValueError(f"Unsupported content type: {response_content_type}")
    
    except Exception as e:
        logger.error(f"Error in output_fn: {str(e)}")
        logger.exception("Stack trace:")
        raise