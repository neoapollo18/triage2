import boto3
import json
import time

# Initialize SageMaker client
sagemaker_client = boto3.client('sagemaker')
runtime_client = boto3.client('sagemaker-runtime')

# Define variables
model_name = '3pl-matching-model'
endpoint_config_name = '3pl-matching-config-large'  # Use a new config name
endpoint_name = '3pl-matching-endpoint'

# Your IAM role with SageMaker permissions
role_arn = 'arn:aws:iam::432774451428:role/3pl-sagemaker-execution-role'  

# S3 path to your model package
model_data_url = 's3://3plsagemaker/model-package.tar.gz'

# Step 1: Create a new endpoint configuration with a more powerful instance
print("\nCreating new endpoint configuration...")
try:
    response = sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'default',
                'ModelName': model_name,  # Using existing model
                'InstanceType': 'ml.c5.2xlarge',  # More powerful instance
                'InitialInstanceCount': 1
            }
        ]
    )
    print(f"New endpoint configuration created: {response['EndpointConfigArn']}")
except Exception as e:
    print(f"Error creating endpoint configuration: {e}")
    exit(1)

# Step 2: Update the existing endpoint to use the new configuration
print("\nUpdating endpoint (this may take several minutes)...")
try:
    response = sagemaker_client.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )
    print(f"Endpoint update initiated: {response['EndpointArn']}")
    
    # Wait for endpoint to be in service
    status = 'Updating'
    while status == 'Updating':
        time.sleep(30)  # Check every 30 seconds
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        print(f"Endpoint status: {status}")
    
    if status == 'InService':
        print("\nEndpoint updated successfully!")
    else:
        print(f"\nEndpoint update failed with status: {status}")
        print(f"Failure reason: {response.get('FailureReason', 'Unknown')}")
except Exception as e:
    print(f"Error updating endpoint: {e}")
    exit(1)

# Optional: Test the endpoint with a simple payload
print("\nWould you like to test the endpoint with a sample payload? (y/n)")
choice = input()

if choice.lower() == 'y':
    try:
        # Very simplified payload
        payload = {
            'business': {
                'business_id': 'BUS123',
                'business_type': 'retail',
                'sku_count': 1000,
                'shipping_volume': 5000,
                'order_speed_expectation': 2.0,
                'log_avg_order_value': 4.5
            },
            '3pl': {
                '3pl_id': '3PL456',
                'headquarters_state': 'CA',
                'min_monthly_volume': 5000,
                'covered_states': 'CA'
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

print("\nDeployment process completed.")