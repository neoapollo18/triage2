import torch
import pickle
import json
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the specific module path to avoid __main__ namespace
from _3pl_matching_model import FeatureEncoder, MatchingModel

def save_model_components():
    """
    Instead of using pickle directly, extract and save the necessary
    model components separately in a deployment-friendly way.
    """
    try:
        # Load the original model
        logger.info("Loading original model checkpoint...")
        checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))
        
        # Extract components
        model_state = checkpoint["model_state_dict"]
        encoder_state = checkpoint["encoder_state"]
        meta_info = {
            "epoch": checkpoint.get("epoch", 0),
            "best_auc": checkpoint.get("best_auc", 0.0)
        }
        
        # Save model state as pure tensor data
        logger.info("Saving model state dict...")
        torch.save(model_state, "model_state.pt")
        
        # Save encoder components individually
        logger.info("Saving encoder components...")
        
        # Extract all encoder attributes we need to save
        encoder_data = {
            "num_bus": encoder_state.num_bus,
            "cat_bus": encoder_state.cat_bus,
            "bin_bus": encoder_state.bin_bus,
            "num_tpl": encoder_state.num_tpl,
            "cat_tpl": encoder_state.cat_tpl,
            "bin_tpl": encoder_state.bin_tpl,
            "states": encoder_state.states,
            "state_to_idx": encoder_state.state_to_idx,
            "vocab_bus": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in encoder_state.vocab_bus.items()},
            "vocab_tpl": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in encoder_state.vocab_tpl.items()}
        }
        
        # Save encoder data as JSON for easy loading
        with open("encoder_data.json", "w") as f:
            json.dump(encoder_data, f)
            
        # Save meta information
        with open("model_meta.json", "w") as f:
            json.dump(meta_info, f)
            
        logger.info("Model components saved successfully!")
        logger.info(f"Model saved at epoch: {meta_info['epoch']}")
        logger.info(f"Best validation AUC: {meta_info['best_auc']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving model components: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = save_model_components()
    sys.exit(0 if success else 1)
