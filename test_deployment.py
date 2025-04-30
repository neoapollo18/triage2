import torch
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting test deployment simulation")

# Import FeatureEncoder first to make it available for model loading
try:
    logger.info("Importing required classes")
    from _3pl_matching_model import FeatureEncoder, MatchingModel
    
    # Add FeatureEncoder to the safe globals list for PyTorch 2.6+ security
    try:
        # PyTorch 2.6+ way
        logger.info("Attempting to register FeatureEncoder as a safe global")
        from torch.serialization import add_safe_globals
        add_safe_globals([('__main__', 'FeatureEncoder')])
        logger.info("Successfully registered FeatureEncoder as safe")
    except ImportError:
        # Fallback for older PyTorch versions
        logger.info("add_safe_globals not available (using older PyTorch version)")
        pass
        
    # Load model checkpoint
    logger.info("Attempting to load model checkpoint")
    try:
        # Try with weights_only=False for PyTorch 2.6+
        logger.info("Trying with weights_only=False")
        checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'), weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't have weights_only parameter
        logger.info("Trying without weights_only parameter")
        checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

    # Extract model and encoder
    logger.info("Extracting model state and encoder")
    model_state = checkpoint["model_state_dict"]
    encoder = checkpoint["encoder_state"]
    
    # Build the model
    logger.info("Rebuilding model architecture")
    model = MatchingModel(encoder)
    model.load_state_dict(model_state)
    model.eval()
    
    logger.info("Model successfully loaded and initialized!")
    logger.info(f"Model was saved at epoch: {checkpoint.get('epoch')}")
    logger.info(f"Best validation AUC: {checkpoint.get('best_auc')}")
    
    print("\n✅ SUCCESS: Model loading simulation completed successfully!")
    
except Exception as e:
    import traceback
    logger.error("Test failed with error:")
    logger.error(str(e))
    logger.error(traceback.format_exc())
    print("\n❌ FAILURE: Model loading simulation failed!")
    sys.exit(1)
