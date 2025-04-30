import torch
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_main_module_fix():
    """Test if our __main__ module fix works for loading the model"""
    try:
        # Import the model classes first
        logger.info("Importing necessary classes")
        from _3pl_matching_model import FeatureEncoder, MatchingModel
        
        # Register FeatureEncoder in the __main__ module
        logger.info("Registering FeatureEncoder in __main__ module")
        import __main__
        __main__.FeatureEncoder = FeatureEncoder
        
        # Now try to load the model
        logger.info("Attempting to load model with __main__ module fix")
        try:
            # Try PyTorch 2.6+ approach with weights_only=False
            checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'), weights_only=False)
            logger.info("Model loaded successfully with weights_only=False parameter")
        except TypeError:
            # Fall back to older PyTorch versions without weights_only parameter
            checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))
            logger.info("Model loaded successfully with basic method")
        
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
        
        print("\n✅ SUCCESS: Model loaded correctly with __main__ module fix!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing __main__ module fix: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print("\n❌ FAILURE: __main__ module fix did not work!")
        return False

if __name__ == "__main__":
    success = test_main_module_fix()
    sys.exit(0 if success else 1)
