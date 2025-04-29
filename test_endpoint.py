import requests

url = "https://triage2-production-ac73.up.railway.app/predict"  # Your deployed API endpoint

payload = {
  "business": {
    "business_id": "bus123",
    "business_type": "ecommerce",
    "target_market": "consumer",
    "sku_count": 150,
    "shipping_volume": 500,
    "order_speed_expectation": 2.5,
    "log_avg_order_value": 4.5,
    "daily_order_variance": 0.3,
    "return_rate_pct": 5.2,
    "avg_sku_turnover_days": 30,
    "avg_package_weight_kg": 1.2,
    "year_founded": 2018,
    "business_age_yrs": 7,
    "growth_velocity_pct": 15,
    "temperature_control_needed": "no",
    "dimensional_weight_class": "light",
    "top_shipping_regions": "NY:0.98;NC:0.02",
    "industry_Fashion": 1,
    "industry_Electronics": 0,
    "industry_Home": 0,
    "industry_Food": 0,
    "industry_Beauty": 0,
    "industry_Health": 0,
    "tech_Shopify": 1,
    "tech_WooCommerce": 0
  },
  "threepl": {
    "min_monthly_volume": 5000,
    "max_monthly_volume": 50000,
    "average_shipping_time_days": 1,
    "dock_to_stock_hours": 12,
    "max_daily_orders": 10000,
    "picking_accuracy_pct": 99.8,
    "available_storage_sqft": 250000,
    "headquarters_state": "CA",
    "num_warehouses": 8,
    "service_coverage": "CA;NY;NV;AZ",
    "covered_states": "CA;NY;NV;AZ"
  }
}

response = requests.post(url, json=payload)

print("Status:", response.status_code)
print("Result:", response.json())