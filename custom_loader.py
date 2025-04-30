import torch
import pickle
import io
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our module - must be done in this file
from _3pl_matching_model import FeatureEncoder, MatchingModel

# Create a very simplified custom loader just for this model
def load_model(model_path):
    """Load the model with special handling for custom classes"""
    
    # Customized class lookup function
    def _lookup_module(module_name):
        if module_name == '__main__':
            # Return our current module instead
            logger.info(f"Redirecting __main__ to custom_loader")
            return sys.modules[__name__]
        else:
            # Normal module lookup
            import importlib
            return importlib.import_module(module_name)
    
    # Save the original _load_module function
    original_load_module = pickle._getattribute(pickle, '_Unpickler', 'find_class')
    
    # Replace it with our custom function
    def _custom_find_class(self, module_name, name):
        module = _lookup_module(module_name)
        if module is None:
            raise ImportError(f"No module named {module_name}")
        return getattr(module, name)
    
    # Apply our patch
    pickle.Unpickler.find_class = _custom_find_class
    
    try:
        # Load with our patched function
        with open(model_path, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')
            logger.info(f"Successfully loaded checkpoint with keys: {list(checkpoint.keys())}")
        
        # Extract model and encoder
        model_state = checkpoint["model_state_dict"]
        encoder = checkpoint["encoder_state"]
        
        # Rebuild the model
        model = MatchingModel(encoder)
        model.load_state_dict(model_state)
        model.eval()
        
        return model, encoder, checkpoint
    
    finally:
        # Restore the original function
        pickle.Unpickler.find_class = original_load_module

# Test function
def test():
    try:
        model, encoder, checkpoint = load_model('best_model.pt')
        logger.info(f"Model was saved at epoch: {checkpoint.get('epoch')}")
        logger.info(f"Best validation AUC: {checkpoint.get('best_auc')}")
        print("\n✅ SUCCESS: Model loading test completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print("\n❌ FAILURE: Model loading test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(test())
