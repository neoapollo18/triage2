import torch
import pandas as pd
import numpy as np
import logging
import sys
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_deployment_readiness():
    """Verify that the model is ready for deployment with our fixes"""
    try:
        logger.info("===== DEPLOYMENT VERIFICATION TEST =====")
        
        # Import FeatureEncoder and register in main (critical fix)
        from _3pl_matching_model import FeatureEncoder, MatchingModel
        import __main__
        __main__.FeatureEncoder = FeatureEncoder
        
        # Load model
        logger.info("Loading model with __main__ fix")
        checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))
        logger.info(f"Model loaded successfully! (epoch: {checkpoint.get('epoch')}, AUC: {checkpoint.get('best_auc'):.4f})")
        
        # Extract components
        model_state = checkpoint["model_state_dict"]
        encoder = checkpoint["encoder_state"]
        
        # Rebuild model
        model = MatchingModel(encoder)
        model.load_state_dict(model_state)
        model.eval()
        
        # Check encoder features
        logger.info("\nFeature verification:")
        logger.info(f"Business numeric features: {encoder.num_bus}")
        logger.info(f"Business categorical features: {encoder.cat_bus}")
        logger.info(f"Business binary features (sample): {encoder.bin_bus[:5] if len(encoder.bin_bus) > 5 else encoder.bin_bus}")
        logger.info(f"3PL numeric features: {encoder.num_tpl}")
        logger.info(f"3PL categorical features: {encoder.cat_tpl}")
        logger.info(f"3PL binary features (sample): {encoder.bin_tpl[:5] if len(encoder.bin_tpl) > 5 else encoder.bin_tpl}")
        
        # Test with simplified input data
        logger.info("\nTesting with simplified API input")
        api_input = {
            "business": {
                "business_id": "test-business",
                "business_type": "ecommerce",
                "target_market": "consumer",
                "temperature_control_needed": "none",
                "dimensional_weight_class": "medium",
                "sku_count": 500, 
                "shipping_volume": 2000,
                "order_speed_expectation": 2,
                "log_avg_order_value": 4.2,
                "daily_order_variance": 0.3,
                "return_rate_pct": 4.5,
                "avg_sku_turnover_days": 20,
                "avg_package_weight_kg": 1.5,
                "year_founded": 2015,
                "business_age_yrs": 10,
                "growth_velocity_pct": 12,
                "top_shipping_regions": "NY:0.6;CA:0.4"
            },
            "threepl": {
                "3pl_id": "test-3pl",
                "headquarters_state": "CA",
                "service_coverage": "nationwide",
                "covered_states": "NY;CA;TX;FL",
                "min_monthly_volume": 1000,
                "max_monthly_volume": 10000,
                "average_shipping_time_days": 3.5,
                "dock_to_stock_hours": 48,
                "max_daily_orders": 500,
                "picking_accuracy_pct": 99.2,
                "available_storage_sqft": 25000,
                "num_warehouses": 2
            }
        }
        
        # Convert to DataFrames
        bus_df = pd.DataFrame([api_input["business"]])
        tpl_df = pd.DataFrame([api_input["threepl"]])
        
        # Ensure all required columns exist with proper data types
        logger.info("Preparing data with proper types and handling missing features")
        
        # Handle numeric columns
        for col in encoder.num_bus:
            if col not in bus_df.columns:
                logger.warning(f"Missing numeric feature: {col}, filling with 0")
                bus_df[col] = 0
            bus_df[col] = pd.to_numeric(bus_df[col], errors='coerce').fillna(0)
            
        for col in encoder.num_tpl:
            if col not in tpl_df.columns:
                logger.warning(f"Missing numeric feature: {col}, filling with 0")
                tpl_df[col] = 0
            tpl_df[col] = pd.to_numeric(tpl_df[col], errors='coerce').fillna(0)
        
        # Handle categorical columns
        for col in encoder.cat_bus:
            if col not in bus_df.columns:
                logger.warning(f"Missing categorical feature: {col}, filling with 'unknown'")
                bus_df[col] = "unknown"
            bus_df[col] = bus_df[col].astype(str).fillna('unknown')
            
        for col in encoder.cat_tpl:
            if col not in tpl_df.columns:
                logger.warning(f"Missing categorical feature: {col}, filling with 'unknown'")
                tpl_df[col] = "unknown"
            tpl_df[col] = tpl_df[col].astype(str).fillna('unknown')
        
        # Handle binary columns
        for col in encoder.bin_bus:
            if col not in bus_df.columns:
                logger.warning(f"Missing binary feature: {col}, filling with 0")
                bus_df[col] = 0
            bus_df[col] = pd.to_numeric(bus_df[col], errors='coerce').fillna(0).astype(int)
                
        for col in encoder.bin_tpl:
            if col not in tpl_df.columns:
                logger.warning(f"Missing binary feature: {col}, filling with 0")
                tpl_df[col] = 0
            tpl_df[col] = pd.to_numeric(tpl_df[col], errors='coerce').fillna(0).astype(int)
        
        # Custom encoding with safety checks - same approach as main.py
        logger.info("Using custom safe encoding approach")
        try:
            # Business features with safety checks
            # For numeric features
            bus_num_values = encoder.scaler_bus.transform(bus_df[encoder.num_bus])
            
            # Add shipping weights
            shipping_weights = np.array([encoder._parse_shipping_regions(row['top_shipping_regions']) 
                                      for _, row in bus_df.iterrows()])
            bus_num = torch.tensor(np.hstack([bus_num_values, shipping_weights]), dtype=torch.float32)
            
            # For binary features
            bus_bin = torch.tensor(bus_df[encoder.bin_bus].values, dtype=torch.float32)
            
            # For categorical features - with safety checks
            # Make sure to get the numpy array from the encoder output 
            cat_values = encoder.enc_bus.transform(bus_df[encoder.cat_bus])
            if hasattr(cat_values, 'values'):  # If it's a DataFrame
                cat_values = cat_values.values
            cat_values = cat_values.astype(int)
            
            # Now verify each categorical value is within embedding range
            for i, col in enumerate(encoder.cat_bus):
                vocab_size = encoder.vocab_bus[col]
                # Ensure values don't exceed vocab size - 1
                cat_values[:, i] = np.minimum(cat_values[:, i], vocab_size - 1)
            
            bus_cat = torch.tensor(cat_values, dtype=torch.long)
            
            # 3PL features with safety checks
            tpl_num_values = encoder.scaler_tpl.transform(tpl_df[encoder.num_tpl])
            
            # Add covered states
            covered_states = np.array([encoder._parse_covered_states(row['covered_states']) 
                                    for _, row in tpl_df.iterrows()])
            tpl_num = torch.tensor(np.hstack([tpl_num_values, covered_states]), dtype=torch.float32)
            
            # For binary features
            tpl_bin = torch.tensor(tpl_df[encoder.bin_tpl].values, dtype=torch.float32)
            
            # For categorical features - with safety checks 
            cat_values = encoder.enc_tpl.transform(tpl_df[encoder.cat_tpl])
            if hasattr(cat_values, 'values'):  # If it's a DataFrame
                cat_values = cat_values.values
            cat_values = cat_values.astype(int)
            
            # Verify each categorical value is within embedding range
            for i, col in enumerate(encoder.cat_tpl):
                vocab_size = encoder.vocab_tpl[col]
                # Ensure values don't exceed vocab size - 1
                cat_values[:, i] = np.minimum(cat_values[:, i], vocab_size - 1)
                
            tpl_cat = torch.tensor(cat_values, dtype=torch.long)
            
            logger.info("Feature encoding successful with safety checks")
        except Exception as e:
            logger.error(f"Error during feature encoding: {e}")
            raise
        
        # Reshape tensors
        bus_data = (bus_num, bus_cat, bus_bin)
        tpl_data = (tpl_num, tpl_cat, tpl_bin)
        
        # Generate prediction
        logger.info("Generating prediction")
        with torch.no_grad():
            match_score = model(bus_data, tpl_data).item()
        
        # Format API response
        response = {
            "match_score": float(match_score),
            "business_id": api_input["business"].get("business_id", "unknown"),
            "3pl_id": api_input["threepl"].get("3pl_id", "unknown")
        }
        
        logger.info(f"Prediction result: match_score={response['match_score']:.4f}")
        
        print("\n✅ SUCCESS: Deployment verification passed! Model should work correctly in production.")
        
        # Output a summary for Railway deployment
        print("\n===== DEPLOYMENT SUMMARY =====")
        print(f"Model epoch: {checkpoint.get('epoch')}")
        print(f"Best validation AUC: {checkpoint.get('best_auc'):.4f}")
        print(f"Example prediction: {response['match_score']:.4f}")
        print("Fixes implemented:")
        print("1. Added FeatureEncoder to __main__ module namespace")
        print("2. Enhanced feature handling to match exact column names")
        print("3. Improved error handling for pickle deserialization")
        print("The model should now work correctly when deployed to Railway!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in deployment verification: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print("\n❌ FAILURE: Deployment verification failed!")
        return False

if __name__ == "__main__":
    success = verify_deployment_readiness()
    sys.exit(0 if success else 1)
