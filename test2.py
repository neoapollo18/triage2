import boto3
import json
import time
import os
from package_model import create_model_package

# Initialize SageMaker client
sagemaker_client = boto3.client('sagemaker')
runtime_client = boto3.client('sagemaker-runtime')

# Create a timestamp for unique naming
timestamp = int(time.time())

# Define variables with timestamp to ensure uniqueness
model_name = f'3pl-matching-model-test-{timestamp}'
endpoint_config_name = f'3pl-matching-config-test-{timestamp}'
endpoint_name = f'3pl-matching-endpoint-test-{timestamp}'

# Your IAM role with SageMaker permissions
role_arn = 'arn:aws:iam::432774451428:role/3pl-sagemaker-execution-role'  

# Package and upload the model, getting the S3 path
print("Packaging and uploading model...")
model_data_url = create_model_package()
print(f"Model package URL: {model_data_url}")

# Step 1: Create a SageMaker model with increased timeouts
print("Creating test SageMaker model...")
try:
    response = sagemaker_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={
            'Image': '763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:1.13.1-cpu-py39-ubuntu20.04-sagemaker',
            'ModelDataUrl': model_data_url,
            'Environment': {
                'SAGEMAKER_PROGRAM': 'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': model_data_url,
                'MODEL_SERVER_TIMEOUT': '900',
                'MODEL_SERVER_WORKERS': '1',
                'LOG_LEVEL': 'DEBUG'
            }
        }
    )
    print(f"Test model created: {response['ModelArn']}")
except Exception as e:
    print(f"Error creating model: {e}")
    exit(1)

# Step 2: Create an endpoint configuration with increased timeouts
print("\nCreating test endpoint configuration...")
try:
    response = sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'default',
                'ModelName': model_name,
                'InstanceType': 'ml.c5.2xlarge',  # More powerful instance
                'InitialInstanceCount': 1,
                'ModelDataDownloadTimeoutInSeconds': 1800,
                'ContainerStartupHealthCheckTimeoutInSeconds': 600
            }
        ]
    )
    print(f"Test endpoint configuration created: {response['EndpointConfigArn']}")
except Exception as e:
    print(f"Error creating endpoint configuration: {e}")
    exit(1)

# Step 3: Create and deploy an endpoint
print("\nDeploying test endpoint (this may take several minutes)...")
try:
    response = sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )
    print(f"Test endpoint creation initiated: {response['EndpointArn']}")

    # Wait for endpoint to be in service
    status = 'Creating'
    while status == 'Creating':
        time.sleep(30)  # Check every 30 seconds
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        print(f"Test endpoint status: {status}")

    if status == 'InService':
        print("\nTest endpoint deployed successfully!")
    else:
        print(f"\nTest endpoint deployment failed with status: {status}")
        print(f"Failure reason: {response.get('FailureReason', 'Unknown')}")
except Exception as e:
    print(f"Error creating endpoint: {e}")
    exit(1)

# Test the endpoint with a more complete payload
print("\nTesting the endpoint with a sample payload...")
try:
    # Complete payload with all required fields
    payload = {
        'business': {
            'business_id': 'BUS123',
            'business_type': 'ecommerce',
            'sku_count': 1500,
            'shipping_volume': 5000,
            'order_speed_expectation': 2.0,
            'log_avg_order_value': 4.5,
            'daily_order_variance': 0.2,
            'return_rate_pct': 3.5,
            'avg_sku_turnover_days': 30,
            'avg_package_weight_kg': 2.0,
            'year_founded': 2015,
            'business_age_yrs': 10,
            'growth_velocity_pct': 15.0,
            'target_market': 'consumer',
            'temperature_control_needed': 'no',
            'dimensional_weight_class': 'light',
            'top_shipping_regions': 'CA:0.4;NY:0.3;TX:0.2;FL:0.1'
        },
        '3pl': {
            '3pl_id': '3PL456',
            'headquarters_state': 'CA',
            'service_coverage': 'national',
            'min_monthly_volume': 5000,
            'max_monthly_volume': 50000,
            'average_shipping_time_days': 2.5,
            'dock_to_stock_hours': 24,
            'max_daily_orders': 2000,
            'picking_accuracy_pct': 99.5,
            'available_storage_sqft': 50000,
            'num_warehouses': 3,
            'covered_states': 'CA;NY;TX;FL'
        }
    }

    print("Invoking endpoint...")
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )

    result = json.loads(response['Body'].read().decode())
    print(f"\nResult: {json.dumps(result, indent=2)}")
except Exception as e:
    print(f"Error testing endpoint: {e}")

print("\nTest deployment process completed.")
print(f"Created endpoint: {endpoint_name}")
print(f"You can delete this endpoint when done testing with:")
print(f"aws sagemaker delete-endpoint --endpoint-name {endpoint_name}")