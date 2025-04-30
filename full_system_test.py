import torch
import pandas as pd
import numpy as np
import pickle
import json
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_pickle_serialization():
    """Test pickle serialization and deserialization"""
    try:
        logger.info("\n===== PICKLE SERIALIZATION TEST =====")
        
        # Import FeatureEncoder and register in main
        from _3pl_matching_model import FeatureEncoder
        import __main__
        __main__.FeatureEncoder = FeatureEncoder
        
        # Load the model to get encoder
        logger.info("Loading model to get encoder")
        checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))
        encoder = checkpoint["encoder_state"]
        
        # Test pickle serialization and deserialization
        logger.info("Testing pickle serialization")
        pickle_data = pickle.dumps(encoder)
        logger.info(f"Pickled data size: {len(pickle_data)} bytes")
        
        # Test deserialization
        logger.info("Testing pickle deserialization")
        encoder_from_pickle = pickle.loads(pickle_data)
        
        # Verify key attributes
        for attr in ['num_bus', 'cat_bus', 'bin_bus', 'num_tpl', 'cat_tpl', 'bin_tpl']:
            if not hasattr(encoder_from_pickle, attr):
                logger.error(f"Missing attribute after unpickling: {attr}")
                return False
        
        logger.info("Pickle serialization test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Error in pickle serialization test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("Pickle serialization test FAILED")
        return False

def test_model_loading():
    """Test model loading with __main__ module fix"""
    try:
        logger.info("\n===== MODEL LOADING TEST =====")
        
        # Import the model classes
        from _3pl_matching_model import FeatureEncoder, MatchingModel
        
        # Register FeatureEncoder in the __main__ module
        logger.info("Registering FeatureEncoder in __main__ module")
        import __main__
        __main__.FeatureEncoder = FeatureEncoder
        
        # Now try to load the model
        logger.info("Loading model checkpoint")
        checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))
        
        # Extract model components
        logger.info("Extracting model components")
        model_state = checkpoint["model_state_dict"]
        encoder = checkpoint["encoder_state"]
        
        # Rebuild the model
        logger.info("Rebuilding model")
        model = MatchingModel(encoder)
        model.load_state_dict(model_state)
        model.eval()
        
        # Check important attributes
        logger.info("Checking model attributes")
        if not hasattr(model, 'business_tower') or not hasattr(model, 'threepl_tower'):
            logger.error("Model missing required attributes")
            return False, None, None
        
        logger.info(f"Model was saved at epoch: {checkpoint.get('epoch')}")
        logger.info(f"Best validation AUC: {checkpoint.get('best_auc')}")
        
        logger.info("Model loading test PASSED")
        return True, model, encoder
        
    except Exception as e:
        logger.error(f"Error in model loading test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("Model loading test FAILED")
        return False, None, None

def test_model_inference(model, encoder):
    """Test model inference with sample data"""
    try:
        logger.info("\n===== MODEL INFERENCE TEST =====")
        
        # Create sample data
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
        
        # Convert to DataFrame
        logger.info("Converting data to DataFrames")
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
        
        logger.info(f"Match score: {match_score:.4f}")
        
        # Check if score is valid
        if 0 <= match_score <= 1:
            logger.info("Match score is valid")
        else:
            logger.error(f"Invalid match score: {match_score}")
            return False
        
        logger.info("Model inference test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Error in model inference test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("Model inference test FAILED")
        return False

def test_api_simulation():
    """Simulate the API flow similar to what happens in FastAPI"""
    try:
        logger.info("\n===== API SIMULATION TEST =====")
        
        # Import the model classes
        from _3pl_matching_model import FeatureEncoder, MatchingModel
        
        # Register FeatureEncoder in the __main__ module
        import __main__
        __main__.FeatureEncoder = FeatureEncoder
        
        # Load model
        logger.info("Loading model in API simulation")
        checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))
        model_state = checkpoint["model_state_dict"]
        encoder = checkpoint["encoder_state"]
        
        # Set up model
        model = MatchingModel(encoder)
        model.load_state_dict(model_state)
        model.eval()
        
        # Create API input
        api_input = {
            "business": {
                "business_id": "api-test",
                "business_type": "retail",
                "shipping_volume": 3000
            },
            "threepl": {
                "3pl_id": "api-tpl",
                "service_coverage": "northeast",
                "min_monthly_volume": 2000
            }
        }
        
        # Convert to DataFrames
        logger.info("Processing API input")
        bus_df = pd.DataFrame([api_input["business"]])
        tpl_df = pd.DataFrame([api_input["threepl"]])
        
        # Fill missing values
        for col in encoder.num_bus:
            if col not in bus_df.columns:
                bus_df[col] = 0
                
        for col in encoder.cat_bus:
            if col not in bus_df.columns:
                bus_df[col] = 'unknown'
                
        for col in encoder.bin_bus:
            if col not in bus_df.columns:
                bus_df[col] = 0
                
        for col in encoder.num_tpl:
            if col not in tpl_df.columns:
                tpl_df[col] = 0
                
        for col in encoder.cat_tpl:
            if col not in tpl_df.columns:
                tpl_df[col] = 'unknown'
                
        for col in encoder.bin_tpl:
            if col not in tpl_df.columns:
                tpl_df[col] = 0
        
        # Add shipping regions and covered states
        if 'top_shipping_regions' not in bus_df.columns:
            bus_df['top_shipping_regions'] = "NY:0.8;MA:0.2"
            
        if 'covered_states' not in tpl_df.columns:
            tpl_df['covered_states'] = "NY;NJ;CT;MA"
        
        # Encode features
        logger.info("Encoding API input")
        bus_num, bus_cat, bus_bin = encoder.encode_business(bus_df)
        tpl_num, tpl_cat, tpl_bin = encoder.encode_3pl(tpl_df)
        
        # Reshape tensors
        bus_data = (bus_num, bus_cat, bus_bin)
        tpl_data = (tpl_num, tpl_cat, tpl_bin)
        
        # Generate prediction
        logger.info("Generating API prediction")
        with torch.no_grad():
            match_score = model(bus_data, tpl_data).item()
        
        # Format API response
        response = {
            "match_score": float(match_score),
            "business_id": api_input["business"].get("business_id", "unknown"),
            "3pl_id": api_input["threepl"].get("3pl_id", "unknown")
        }
        
        logger.info(f"API response: {json.dumps(response, indent=2)}")
        
        logger.info("API simulation test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Error in API simulation test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("API simulation test FAILED")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("==================================================")
    print("FULL SYSTEM TEST FOR 3PL MATCHING MODEL DEPLOYMENT")
    print("==================================================")
    
    test_results = {}
    
    # Test 1: Pickle Serialization
    pickle_success = test_pickle_serialization()
    test_results["pickle_serialization"] = pickle_success
    
    # Test 2: Model Loading
    model_success, model, encoder = test_model_loading()
    test_results["model_loading"] = model_success
    
    # Only continue if model loading succeeded
    if model_success and model is not None and encoder is not None:
        # Test 3: Model Inference
        inference_success = test_model_inference(model, encoder)
        test_results["model_inference"] = inference_success
    else:
        test_results["model_inference"] = False
    
    # Test 4: API Simulation
    api_success = test_api_simulation()
    test_results["api_simulation"] = api_success
    
    # Print summary
    print("\n==================================================")
    print("TEST RESULTS SUMMARY")
    print("==================================================")
    
    all_passed = True
    for test, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test.replace('_', ' ').title()}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED - System is ready for deployment!")
    else:
        print("\n❌ SOME TESTS FAILED - Please review issues before deploying.")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
