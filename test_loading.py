import logging
import sys
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pytorch_loading():
    try:
        # Add current directory to path
        sys.path.insert(0, ".")
        
        # Import the model classes
        logger.info("Importing model classes")
        from _3pl_matching_model import FeatureEncoder, MatchingModel
        
        # Register classes if possible
        try:
            import torch.serialization
            # Register the FeatureEncoder class
            logger.info("Attempting to register FeatureEncoder for safe loading")
            torch.serialization._get_module("_3pl_matching_model")
            torch.serialization._get_module("__main__")
            pickle_global = torch.serialization._get_global("_3pl_matching_model", "FeatureEncoder")
            torch.serialization._add_global_to_namespace(pickle_global, FeatureEncoder)
            logger.info("Successfully registered FeatureEncoder")
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not register classes for safe loading: {e}")
            logger.warning("Will try loading without registration")
        
        # Try loading the model
        logger.info("Attempting to load model checkpoint")
        try:
            # First try with weights_only=False
            checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'), weights_only=False)
            logger.info("✅ Model loaded successfully with weights_only=False")
        except TypeError as e:
            logger.info(f"TypeError with weights_only: {e}")
            # Fallback for older PyTorch
            checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))
            logger.info("✅ Model loaded successfully with older PyTorch method")
        
        # Extract model components
        logger.info("Extracting model components")
        model_state = checkpoint["model_state_dict"]
        encoder = checkpoint["encoder_state"]
        
        # Rebuild the model
        logger.info("Rebuilding model")
        model = MatchingModel(encoder)
        model.load_state_dict(model_state)
        model.eval()
        
        # Print model details
        logger.info(f"Model was saved at epoch: {checkpoint.get('epoch')}")
        logger.info(f"Best validation AUC: {checkpoint.get('best_auc')}")
        
        print("\n✅ SUCCESS: Model loading test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing model loading: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print("\n❌ FAILURE: Model loading test failed!")
        return False

if __name__ == "__main__":
    success = test_pytorch_loading()
    sys.exit(0 if success else 1)
