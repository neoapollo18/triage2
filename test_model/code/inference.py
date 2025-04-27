# test_model/code/inference.py
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

def model_fn(model_dir):
    """Dummy model function for testing."""
    try:
        logger.info(f"TEST MODE: Starting model loading from {model_dir}")
        
        # Return a dummy model
        logger.info("TEST MODE: Created dummy model successfully")
        return {"dummy_model": True}
    except Exception as e:
        logger.error(f"Error in model_fn: {str(e)}")
        raise

def input_fn(request_body, request_content_type):
    """Parse input data."""
    try:
        logger.info(f"TEST MODE: Processing input with content type: {request_content_type}")
        if request_content_type == 'application/json':
            return json.loads(request_body)
        else:
            return request_body
    except Exception as e:
        logger.error(f"Error in input_fn: {str(e)}")
        raise

def predict_fn(input_data, model):
    """Generate a simple prediction."""
    try:
        logger.info("TEST MODE: Making prediction with dummy model")
        return {
            "status": "success",
            "message": "This is a test endpoint response",
            "input_received": str(input_data)[:100]
        }
    except Exception as e:
        logger.error(f"Error in predict_fn: {str(e)}")
        raise

def output_fn(prediction, response_content_type):
    """Format the prediction output."""
    try:
        logger.info("TEST MODE: Formatting output")
        if response_content_type == 'application/json':
            return json.dumps(prediction)
        else:
            return str(prediction)
    except Exception as e:
        logger.error(f"Error in output_fn: {str(e)}")
        raise