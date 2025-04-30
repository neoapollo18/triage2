import torch
import pickle
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our model classes
from _3pl_matching_model import FeatureEncoder, MatchingModel

# Create module mapping for pickle
class _ModuleMapping:
    def __init__(self):
        self._module_cache = {}
    
    def __call__(self, module_name):
        # Map __main__ to _3pl_matching_model for classes like FeatureEncoder
        if module_name == "__main__":
            logger.info(f"Redirecting {module_name} to _3pl_matching_model module")
            return sys.modules["_3pl_matching_model"]
        # Default behavior
        if module_name not in self._module_cache:
            self._module_cache[module_name] = __import__(module_name, fromlist=['object'])
        return self._module_cache[module_name]

def test_module_mapping():
    try:
        # Create module resolver
        logger.info("Creating module resolver mapping")
        module_resolver = _ModuleMapping()
        
        # Try to load with module mapping
        logger.info("Attempting to load model with module mapping")
        try:
            # Try with weights_only=False (PyTorch 2.6+)
            checkpoint = torch.load(
                "best_model.pt", 
                map_location=torch.device('cpu'),
                weights_only=False,
                pickle_module=pickle,
                pickle_load_args={"module_mapping": module_resolver}
            )
            logger.info("Model loaded successfully with weights_only=False")
        except (TypeError, AttributeError) as e:
            logger.info(f"Error with first attempt: {e}")
            
            # Try without weights_only (older PyTorch)
            try:
                checkpoint = torch.load(
                    "best_model.pt",
                    map_location=torch.device('cpu'),
                    pickle_module=pickle,
                    pickle_load_args={"module_mapping": module_resolver}
                )
                logger.info("Model loaded successfully with older PyTorch method")
            except Exception as e:
                logger.info(f"Error with second attempt: {e}")
                
                # Last resort - try basic loading
                checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))
                logger.info("Model loaded with basic method")
        
        # Extract model components
        logger.info("Extracting model components")
        model_state = checkpoint["model_state_dict"]
        encoder = checkpoint["encoder_state"]
        
        # Rebuild model
        logger.info("Rebuilding model")
        model = MatchingModel(encoder)
        model.load_state_dict(model_state)
        model.eval()
        
        # Print model details
        logger.info(f"Model was saved at epoch: {checkpoint.get('epoch')}")
        logger.info(f"Best validation AUC: {checkpoint.get('best_auc')}")
        
        print("\n✅ SUCCESS: Module mapping test completed successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error testing module mapping: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print("\n❌ FAILURE: Module mapping test failed!")
        return False

if __name__ == "__main__":
    success = test_module_mapping()
    sys.exit(0 if success else 1)
