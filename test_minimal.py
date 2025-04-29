import requests
import json

url = "https://triage2-production-ac73.up.railway.app/predict"

# Simplified test payload with focus on required fields
payload = {
  "business": {
    "business_id": "bus123",
    "business_type": "B2C",  # Use a value from your training data
    "target_market": "National",  # Use a value from your training data
    "sku_count": 0,
    "shipping_volume": 0,
    "order_speed_expectation": 0,
    "log_avg_order_value": 0,
    "daily_order_variance": 0,
    "return_rate_pct": 0,
    "avg_sku_turnover_days": 0,
    "avg_package_weight_kg": 0,
    "year_founded": 0,
    "business_age_yrs": 0,
    "growth_velocity_pct": 15,
    "temperature_control_needed": "frozen",  # Use a value from your training data
    "dimensional_weight_class": "bulky",  # Use a value from your training data
    "top_shipping_regions": "NY:0.98;NC:0.02"  # This exact format from businesses.csv
  },
  "threepl": {
    "3pl_id": "3pl456",
    "headquarters_state": "TX",
    "service_coverage": "NY;NC",
    "min_monthly_volume": 0,
    "max_monthly_volume": 0,
    "average_shipping_time_days": 0,
    "dock_to_stock_hours": 0,
    "max_daily_orders": 0,
    "picking_accuracy_pct": 0,
    "available_storage_sqft": 0,
    "num_warehouses": 0,
    "covered_states": "NY;NC"
  }
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        print(f"Success! Result: {response.json()}")
    else:
        print(f"Error: {response.json()}")
        
except Exception as e:
    print(f"Request failed: {str(e)}")
