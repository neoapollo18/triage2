import os
import tarfile
import boto3
import time

def create_model_package():
    # Create a timestamp for unique model naming
    timestamp = int(time.time())
    package_name = f"model-package-{timestamp}.tar.gz"
    
    # Make sure directory structure exists
    os.makedirs('deployment/code', exist_ok=True)
    os.makedirs('deployment/model', exist_ok=True)
    
    # Create a tarfile
    with tarfile.open(package_name, 'w:gz') as tar:
        # Add inference script
        tar.add('deployment/code/inference.py', arcname='inference.py')
        
        # Add model module
        tar.add('deployment/code/model_module.py', arcname='model_module.py')
        
        # Add model file
        tar.add('deployment/model/best_model.pt', arcname='best_model.pt')
        
        # Add requirements file
        tar.add('deployment/code/requirements.txt', arcname='requirements.txt')

    print(f"Model package created at {package_name}")
    
    # Upload to S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(
        package_name, 
        '3plsagemaker',  # Your bucket name
        package_name
    )
    print(f"Model package uploaded to S3: s3://3plsagemaker/{package_name}")
    
    return f"s3://3plsagemaker/{package_name}"

if __name__ == '__main__':
    create_model_package()