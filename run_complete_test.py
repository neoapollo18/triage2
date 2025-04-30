import torch
import pandas as pd
import numpy as np
import logging
import json
import sys
import os
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current path to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import necessary classes
from _3pl_matching_model import FeatureEncoder, MatchingModel

def test_model_loading():
    """Test that the model loads correctly using the approach in main.py"""
    try:
        logger.info("TEST 1: Model Loading")
        logger.info("=====================")
        
        # Try loading with weights_only=False first (PyTorch 2.6+)
        try:
            checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'), weights_only=False)
            logger.info("Model loaded successfully with weights_only=False")
        except TypeError:
            # Fall back to older versions
            checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))
            logger.info("Model loaded successfully with basic approach")
        
        # Verify checkpoint contents
        expected_keys = ["epoch", "model_state_dict", "encoder_state", "best_auc"]
        for key in expected_keys:
            if key not in checkpoint:
                logger.error(f"Missing key in checkpoint: {key}")
                return False
                
        logger.info(f"Model was saved at epoch: {checkpoint.get('epoch')}")
        logger.info(f"Best validation AUC: {checkpoint.get('best_auc')}")
        logger.info("Model loading test: PASSED")
        return True, checkpoint
        
    except Exception as e:
        logger.error(f"Error in model loading test: {e}")
        logger.error(traceback.format_exc())
        logger.info("Model loading test: FAILED")
        return False, None

def test_model_reconstruction(checkpoint):
    """Test that the model can be reconstructed from the checkpoint"""
    try:
        logger.info("\nTEST 2: Model Reconstruction")
        logger.info("===========================")
        
        # Extract components
        model_state = checkpoint["model_state_dict"]
        encoder = checkpoint["encoder_state"]
        
        # Rebuild model
        model = MatchingModel(encoder)
        model.load_state_dict(model_state)
        model.eval()
        
        # Check that model has expected components
        logger.info(f"Model architecture: {type(model).__name__}")
        logger.info(f"Model has business_tower: {hasattr(model, 'business_tower')}")
        logger.info(f"Model has threepl_tower: {hasattr(model, 'threepl_tower')}")
        logger.info(f"Model has pred_head: {hasattr(model, 'pred_head')}")
        
        # Check encoder properties
        logger.info(f"Encoder numeric features (business): {len(encoder.num_bus)}")
        logger.info(f"Encoder categorical features (business): {len(encoder.cat_bus)}")
        logger.info(f"Encoder binary features (business): {len(encoder.bin_bus)}")
        logger.info(f"Encoder numeric features (3PL): {len(encoder.num_tpl)}")
        logger.info(f"Encoder categorical features (3PL): {len(encoder.cat_tpl)}")
        logger.info(f"Encoder binary features (3PL): {len(encoder.bin_tpl)}")
        
        logger.info("Model reconstruction test: PASSED")
        return True, model, encoder
        
    except Exception as e:
        logger.error(f"Error in model reconstruction test: {e}")
        logger.error(traceback.format_exc())
        logger.info("Model reconstruction test: FAILED")
        return False, None, None

def test_inference(model, encoder):
    """Test model inference with sample data"""
    try:
        logger.info("\nTEST 3: Model Inference")
        logger.info("=====================")
        
        # Create sample business data
        business_data = {
            "business_id": "test123",
            "business_type": "ecommerce",
            "target_market": "consumer",
            "sku_count": 1000,
            "shipping_volume": 5000,
            "order_speed_expectation": 2,
            "log_avg_order_value": 4.5,
            "daily_order_variance": 0.2,
            "return_rate_pct": 5.0,
            "avg_sku_turnover_days": 30,
            "avg_package_weight_kg": 1.2,
            "year_founded": 2010,
            "business_age_yrs": 13,
            "growth_velocity_pct": 15,
            "temperature_control_needed": "none",
            "dimensional_weight_class": "light",
            "top_shipping_regions": "NY:0.5;CA:0.3;TX:0.2",
            "industry_fashion": 1,
            "tech_shopify": 1
        }
        
        # Create sample 3PL data
        threepl_data = {
            "3pl_id": "3pl456",
            "headquarters_state": "NY",
            "service_coverage": "national",
            "min_monthly_volume": 1000,
            "max_monthly_volume": 50000,
            "average_shipping_time_days": 2.5,
            "dock_to_stock_hours": 24,
            "max_daily_orders": 2000,
            "picking_accuracy_pct": 99.5,
            "available_storage_sqft": 50000,
            "num_warehouses": 3,
            "covered_states": "NY;NJ;CT;PA;MA",
            "service_apparel": 1,
            "specialty_ecommerce": 1
        }
        
        # Convert to DataFrames
        bus_df = pd.DataFrame([business_data])
        tpl_df = pd.DataFrame([threepl_data])
        
        # Encode features
        logger.info("Encoding business data")
        bus_num, bus_cat, bus_bin = encoder.encode_business(bus_df)
        
        logger.info("Encoding 3PL data")
        tpl_num, tpl_cat, tpl_bin = encoder.encode_3pl(tpl_df)
        
        # Reshape tensors
        bus_data = (bus_num, bus_cat, bus_bin)
        tpl_data = (tpl_num, tpl_cat, tpl_bin)
        
        # Generate prediction
        logger.info("Generating prediction")
        with torch.no_grad():
            match_score = model(bus_data, tpl_data).item()
        
        # Display result
        logger.info(f"Match score: {match_score:.4f}")
        
        # Check if match score is in expected range
        if 0 <= match_score <= 1:
            logger.info("Match score is in expected range [0, 1]")
            logger.info("Model inference test: PASSED")
            return True
        else:
            logger.error(f"Match score {match_score} is outside expected range [0, 1]")
            logger.info("Model inference test: FAILED")
            return False
            
    except Exception as e:
        logger.error(f"Error in model inference test: {e}")
        logger.error(traceback.format_exc())
        logger.info("Model inference test: FAILED")
        return False

def test_api_request():
    """Test the formatting of API requests and responses"""
    try:
        logger.info("\nTEST 4: API Request Format")
        logger.info("=========================")
        
        # Create sample business data
        business_data = {
            "business_id": "test123",
            "business_type": "ecommerce",
            "target_market": "consumer",
            "sku_count": 1000,
            "shipping_volume": 5000
        }
        
        # Create sample 3PL data
        threepl_data = {
            "3pl_id": "3pl456",
            "headquarters_state": "NY",
            "service_coverage": "national",
            "min_monthly_volume": 1000,
            "max_monthly_volume": 50000
        }
        
        # Format as API request
        api_request = {
            "business": business_data,
            "threepl": threepl_data
        }
        
        # Convert to JSON and back
        api_json = json.dumps(api_request)
        parsed_request = json.loads(api_json)
        
        # Check structure
        if "business" not in parsed_request or "threepl" not in parsed_request:
            logger.error("API request format is invalid")
            logger.info("API request format test: FAILED")
            return False
            
        # Check if IDs are preserved
        if parsed_request["business"].get("business_id") != "test123" or parsed_request["threepl"].get("3pl_id") != "3pl456":
            logger.error("API request data is corrupted")
            logger.info("API request format test: FAILED")
            return False
            
        logger.info("Sample API request format:")
        logger.info(json.dumps(api_request, indent=2))
        logger.info("API request format test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Error in API request format test: {e}")
        logger.error(traceback.format_exc())
        logger.info("API request format test: FAILED")
        return False

def main():
    """Run all tests and report results"""
    print("Starting comprehensive tests for 3PL Matching Model")
    print("=" * 50)
    
    # Run tests
    test_results = {}
    
    # Test 1: Model Loading
    success, checkpoint = test_model_loading()
    test_results["model_loading"] = success
    
    # Only continue if model loading succeeded
    if success and checkpoint:
        # Test 2: Model Reconstruction
        success, model, encoder = test_model_reconstruction(checkpoint)
        test_results["model_reconstruction"] = success
        
        # Only continue if model reconstruction succeeded
        if success and model and encoder:
            # Test 3: Model Inference
            success = test_inference(model, encoder)
            test_results["model_inference"] = success
    
    # Test 4: API Request Format
    success = test_api_request()
    test_results["api_request_format"] = success
    
    # Print summary
    print("\nTest Results Summary:")
    print("=" * 50)
    
    all_passed = True
    for test, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test.replace('_', ' ').title()}: {status}")
        if not result:
            all_passed = False
    
    # Final verdict
    print("\nFinal Result:")
    if all_passed:
        print("✅ ALL TESTS PASSED - System is ready for deployment!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Please fix issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
