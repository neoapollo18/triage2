import torch
import sys
import os
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def prepare_model_for_deployment():
    """
    Prepare the model for deployment by explicitly saving components in
    a way that doesn't rely on pickle's module resolution
    """
    try:
        logger.info("Loading model components...")
        
        # First, import properly from a non-main module
        from _3pl_matching_model import FeatureEncoder, MatchingModel
        
        # Try to load the original model
        try:
            logger.info("Loading original model checkpoint")
            checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))
            logger.info("Successfully loaded original model")
        except Exception as e:
            logger.error(f"Failed to load model directly: {e}")
            logger.info("Will attempt manual reconstruction")
            return False
            
        # Extract key components
        model_state = checkpoint["model_state_dict"]
        original_encoder = checkpoint["encoder_state"]
        epoch = checkpoint.get("epoch", 0)
        best_auc = checkpoint.get("best_auc", 0.0)
        
        logger.info(f"Original model epoch: {epoch}, AUC: {best_auc}")
        
        # Create a fresh encoder with the same data
        logger.info("Creating deployment-friendly encoder")
        
        # Create a new encoder instance with the extracted data
        new_encoder = FeatureEncoder(None, None)  # Initialize without data
        
        # Manually copy key attributes from original encoder
        new_encoder.num_bus = original_encoder.num_bus
        new_encoder.cat_bus = original_encoder.cat_bus
        new_encoder.bin_bus = original_encoder.bin_bus
        new_encoder.num_tpl = original_encoder.num_tpl
        new_encoder.cat_tpl = original_encoder.cat_tpl
        new_encoder.bin_tpl = original_encoder.bin_tpl
        new_encoder.states = original_encoder.states
        new_encoder.state_to_idx = original_encoder.state_to_idx
        new_encoder.vocab_bus = original_encoder.vocab_bus
        new_encoder.vocab_tpl = original_encoder.vocab_tpl
        
        # Create a new model with the clean encoder
        logger.info("Creating new model")
        new_model = MatchingModel(new_encoder)
        
        # Load the state dict
        logger.info("Loading model weights")
        new_model.load_state_dict(model_state)
        new_model.eval()
        
        # Create a new checkpoint that will be saved with proper module paths
        logger.info("Creating deployment-friendly checkpoint")
        deployment_checkpoint = {
            "model_state_dict": new_model.state_dict(),
            "encoder_state": new_encoder,
            "epoch": epoch,
            "best_auc": best_auc
        }
        
        # Save the deployment-ready model
        logger.info("Saving deployment-ready model")
        torch.save(deployment_checkpoint, "deployment_model.pt")
        
        logger.info("Model prepared successfully for deployment")
        logger.info(f"Saved to: deployment_model.pt")
        logger.info(f"Epoch: {epoch}, Best AUC: {best_auc}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error preparing model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = prepare_model_for_deployment()
    if success:
        print("\n✅ SUCCESS: Model prepared for deployment!")
    else:
        print("\n❌ FAILURE: Could not prepare model for deployment.")
    sys.exit(0 if success else 1)
