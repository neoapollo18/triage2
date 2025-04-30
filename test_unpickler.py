import torch
import pickle
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import needed classes
from _3pl_matching_model import FeatureEncoder, MatchingModel

def test_custom_unpickler():
    try:
        logger.info("Testing custom unpickler approach")
        
        # Create a proper subclass of Unpickler
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module_name, name):
                # If trying to load FeatureEncoder from __main__, use our imported version
                if module_name == '__main__' and name == 'FeatureEncoder':
                    logger.info(f"Redirecting {module_name}.{name} to imported FeatureEncoder")
                    return FeatureEncoder
                # Otherwise, use the default behavior
                return super().find_class(module_name, name)
        
        # Read the file manually
        with open("best_model.pt", "rb") as f:
            # Create our custom unpickler
            unpickler = CustomUnpickler(f)
            
            # Load the checkpoint
            checkpoint = unpickler.load()
            logger.info("Model loaded successfully with custom unpickler")
        
        # Extract model and encoder
        model_state = checkpoint["model_state_dict"]
        encoder = checkpoint["encoder_state"]
        
        # Rebuild the model
        model = MatchingModel(encoder)
        model.load_state_dict(model_state)
        model.eval()
        
        # Print model details
        logger.info(f"Model was saved at epoch: {checkpoint.get('epoch')}")
        logger.info(f"Best validation AUC: {checkpoint.get('best_auc')}")
        
        print("\n✅ SUCCESS: Custom unpickler approach works correctly!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing custom unpickler: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print("\n❌ FAILURE: Custom unpickler approach failed!")
        return False

if __name__ == "__main__":
    success = test_custom_unpickler()
    sys.exit(0 if success else 1)
