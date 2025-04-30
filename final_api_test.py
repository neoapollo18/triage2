"""
Final API test to verify deployment readiness with our safe encoder approach.
This simulates API requests with different data variations to ensure robust handling.
"""
import torch
import pandas as pd
import numpy as np
import logging
import sys
import os
import json
from fastapi.testclient import TestClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import FeatureEncoder and register in __main__ to avoid unpickling issues
from _3pl_matching_model import FeatureEncoder
import __main__
__main__.FeatureEncoder = FeatureEncoder

def final_api_test():
    """Run a final comprehensive test of the API with various input scenarios"""
    try:
        # Import the FastAPI app - this should succeed with our __main__ fix
        from main import app
        
        # Create test client
        client = TestClient(app)
        
        print("\n" + "="*50)
        print("FINAL API TEST FOR 3PL MATCHING MODEL")
        print("="*50)
        
        # Test cases to verify API robustness
        test_cases = [
            {
                "name": "Minimal valid data",
                "payload": {
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
            },
            {
                "name": "Missing binary features", 
                "payload": {
                    "business": {
                        "business_id": "missing-binary-business",
                        "business_type": "retail",
                        "target_market": "business",
                        "temperature_control_needed": "refrigerated",
                        "dimensional_weight_class": "heavy",
                        "sku_count": 150, 
                        "shipping_volume": 500,
                        "order_speed_expectation": 1,
                        "log_avg_order_value": 6.8,
                        "daily_order_variance": 0.5,
                        "return_rate_pct": 2.0,
                        "avg_sku_turnover_days": 45,
                        "avg_package_weight_kg": 5.0,
                        "year_founded": 2010,
                        "business_age_yrs": 15,
                        "growth_velocity_pct": 8,
                        "top_shipping_regions": "CA:0.7;WA:0.3"
                    },
                    "threepl": {
                        "3pl_id": "missing-binary-3pl",
                        "headquarters_state": "WA",
                        "service_coverage": "regional",
                        "covered_states": "CA;WA;OR",
                        "min_monthly_volume": 500,
                        "max_monthly_volume": 5000,
                        "average_shipping_time_days": 2.0,
                        "dock_to_stock_hours": 24,
                        "max_daily_orders": 200,
                        "picking_accuracy_pct": 99.5,
                        "available_storage_sqft": 15000,
                        "num_warehouses": 1
                    }
                }
            },
            {
                "name": "Unknown categorical values",
                "payload": {
                    "business": {
                        "business_id": "unknown-categories",
                        "business_type": "UNKNOWN_TYPE",  # Unknown value
                        "target_market": "hybrid",  # Unknown value
                        "temperature_control_needed": "special",  # Unknown value
                        "dimensional_weight_class": "oversized",  # Unknown value
                        "sku_count": 1200, 
                        "shipping_volume": 8000,
                        "order_speed_expectation": 3,
                        "log_avg_order_value": 5.1,
                        "daily_order_variance": 0.8,
                        "return_rate_pct": 7.5,
                        "avg_sku_turnover_days": 15,
                        "avg_package_weight_kg": 2.2,
                        "year_founded": 2020,
                        "business_age_yrs": 5,
                        "growth_velocity_pct": 25,
                        "top_shipping_regions": "FL:0.5;TX:0.5"
                    },
                    "threepl": {
                        "3pl_id": "unknown-categories-3pl",
                        "headquarters_state": "FOREIGN",  # Unknown value
                        "service_coverage": "UNKNOWN",  # Unknown value
                        "covered_states": "FL;TX;GA",
                        "min_monthly_volume": 2000,
                        "max_monthly_volume": 20000,
                        "average_shipping_time_days": 4.5,
                        "dock_to_stock_hours": 72,
                        "max_daily_orders": 800,
                        "picking_accuracy_pct": 98.5,
                        "available_storage_sqft": 40000,
                        "num_warehouses": 3
                    }
                }
            },
            {
                "name": "Extreme numeric values",
                "payload": {
                    "business": {
                        "business_id": "extreme-values",
                        "business_type": "ecommerce",
                        "target_market": "consumer",
                        "temperature_control_needed": "none",
                        "dimensional_weight_class": "medium",
                        "sku_count": 100000,  # Extreme high value
                        "shipping_volume": 500000,  # Extreme high value
                        "order_speed_expectation": 10,  # Extreme high value
                        "log_avg_order_value": 10.0,  # Extreme high value
                        "daily_order_variance": 5.0,  # Extreme high value
                        "return_rate_pct": 50.0,  # Extreme high value
                        "avg_sku_turnover_days": 1,  # Extreme low value
                        "avg_package_weight_kg": 50.0,  # Extreme high value
                        "year_founded": 1900,  # Extreme low value
                        "business_age_yrs": 150,  # Extreme high value
                        "growth_velocity_pct": 200,  # Extreme high value
                        "top_shipping_regions": "NY:1.0"  # Single state
                    },
                    "threepl": {
                        "3pl_id": "extreme-values-3pl",
                        "headquarters_state": "CA",
                        "service_coverage": "nationwide",
                        "covered_states": "ALL",  # Invalid format
                        "min_monthly_volume": 0,  # Extreme low value
                        "max_monthly_volume": 1000000,  # Extreme high value
                        "average_shipping_time_days": 0.5,  # Extreme low value
                        "dock_to_stock_hours": 1,  # Extreme low value
                        "max_daily_orders": 10000,  # Extreme high value
                        "picking_accuracy_pct": 100.0,  # Perfect accuracy
                        "available_storage_sqft": 1000000,  # Extreme high value
                        "num_warehouses": 50  # Extreme high value
                    }
                }
            }
        ]
        
        # Run all test cases
        results = []
        for i, test in enumerate(test_cases):
            print(f"\nTest Case {i+1}: {test['name']}")
            print("-" * 40)
            
            try:
                # Call the API endpoint
                response = client.post("/predict", json=test["payload"])
                
                # Check response
                if response.status_code == 200:
                    match_score = response.json().get("match_score", 0)
                    print(f"✅ SUCCESS - Status: {response.status_code}, Match Score: {match_score:.4f}")
                    results.append({
                        "name": test["name"],
                        "status": "PASSED",
                        "code": response.status_code,
                        "match_score": match_score
                    })
                else:
                    print(f"❌ FAILED - Status: {response.status_code}, Response: {response.text}")
                    results.append({
                        "name": test["name"],
                        "status": "FAILED",
                        "code": response.status_code,
                        "response": response.text
                    })
            except Exception as e:
                print(f"❌ ERROR - {str(e)}")
                results.append({
                    "name": test["name"],
                    "status": "ERROR",
                    "error": str(e)
                })
                
        # Print summary
        print("\n" + "="*50)
        print("TEST RESULTS SUMMARY")
        print("="*50)
        
        passed = sum(1 for r in results if r["status"] == "PASSED")
        print(f"Tests Passed: {passed}/{len(test_cases)}")
        
        for result in results:
            status_symbol = "✅" if result["status"] == "PASSED" else "❌"
            print(f"{status_symbol} {result['name']}: {result['status']}")
        
        if passed == len(test_cases):
            print("\n✅ ALL TESTS PASSED - The model and API are ready for deployment!")
            print("The safe encoder approach successfully handles all input variations.")
            return True
        else:
            print(f"\n❌ {len(test_cases) - passed} TESTS FAILED - Please fix the issues before deploying.")
            return False
            
    except Exception as e:
        logger.error(f"Error running final API test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\n❌ SETUP ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    success = final_api_test()
    sys.exit(0 if success else 1)
