import boto3
import json
from botocore.config import Config
import time

# Configure extended timeout
boto_config = Config(
    read_timeout=300,  # 5 minutes timeout
    connect_timeout=300,
    retries={'max_attempts': 0}
)

# Initialize client
runtime_client = boto3.client('sagemaker-runtime', config=boto_config)
endpoint_name = '3pl-matching-endpoint'

print(f"Testing endpoint: {endpoint_name}")
print("Preparing minimal test payload...")

# Extremely simplified test payload
payload = {
    'business': {
        'business_id': 'BUS123',
        'business_type': 'retail',
        'sku_count': 1000,
        'shipping_volume': 5000,
        'order_speed_expectation': 2.0,
        'log_avg_order_value': 4.5,
        'daily_order_variance': 0.2,
        'return_rate_pct': 3.0,
        'avg_sku_turnover_days': 30,
        'avg_package_weight_kg': 1.0,
        'year_founded': 2015,
        'business_age_yrs': 10,
        'growth_velocity_pct': 10.0,
        'top_shipping_regions': 'CA:0.5',
        'target_market': 'direct_to_consumer',
        'temperature_control_needed': 'ambient',
        'dimensional_weight_class': 'light',
        'industry_retail': 1
    },
    '3pl': {
        '3pl_id': '3PL456',
        'headquarters_state': 'CA',
        'service_coverage': 'regional',
        'min_monthly_volume': 1000,
        'max_monthly_volume': 10000,
        'average_shipping_time_days': 2.0,
        'dock_to_stock_hours': 24,
        'max_daily_orders': 500,
        'picking_accuracy_pct': 99.0,
        'available_storage_sqft': 10000,
        'num_warehouses': 2,
        'covered_states': 'CA',
        'tech_wms': 1
    }
}

print(f"Payload prepared: {json.dumps(payload, indent=2)}")
print("Invoking endpoint (this may take a few minutes)...")

try:
    # Record start time
    start_time = time.time()
    
    # Invoke the endpoint
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload),
        Accept='application/json'
    )
    
    # Record end time
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Request completed in {elapsed:.2f} seconds")
    
    # Process response
    response_body = response['Body'].read().decode()
    result = json.loads(response_body)
    print(f"\nResult: {json.dumps(result, indent=2)}")
    
except Exception as e:
    print(f"Error testing endpoint: {e}")
    print("\nFetching the latest logs...")
    
    # Try to get the latest logs
    try:
        logs_client = boto3.client('logs')
        response = logs_client.get_log_events(
            logGroupName='/aws/sagemaker/Endpoints/3pl-matching-endpoint',
            logStreamName='default/i-0f889a8771a90609a',
            limit=20,
            startFromHead=False  # Get the most recent logs
        )
        
        print("\nLatest logs:")
        for event in response['events']:
            print(f"{event['message']}")
    except Exception as log_error:
        print(f"Error fetching logs: {log_error}")

print("\nTest completed.")