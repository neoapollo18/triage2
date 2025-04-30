import torch
import json
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the needed model classes from specific modules (not __main__)
from _3pl_matching_model import FeatureEncoder, MatchingModel

class EncoderStateLoader:
    """A class to recreate the FeatureEncoder from saved data"""
    
    def __init__(self, num_bus, cat_bus, bin_bus, num_tpl, cat_tpl, bin_tpl, 
                 states, state_to_idx, vocab_bus, vocab_tpl):
        self.num_bus = num_bus
        self.cat_bus = cat_bus
        self.bin_bus = bin_bus
        self.num_tpl = num_tpl
        self.cat_tpl = cat_tpl 
        self.bin_tpl = bin_tpl
        self.states = states
        self.state_to_idx = state_to_idx
        self.vocab_bus = {k: torch.tensor(v) if isinstance(v, list) else v 
                          for k, v in vocab_bus.items()}
        self.vocab_tpl = {k: torch.tensor(v) if isinstance(v, list) else v 
                          for k, v in vocab_tpl.items()}
    
    def encode_business(self, data):
        """Process business data into model-compatible tensors"""
        # Implementation matching original FeatureEncoder.encode_business
        # This would contain the same logic as in your original class
        pass
        
    def encode_3pl(self, data):
        """Process 3PL data into model-compatible tensors"""
        # Implementation matching original FeatureEncoder.encode_3pl
        # This would contain the same logic as in your original class
        pass

def load_model():
    """Load model components in a deployment-friendly way"""
    try:
        logger.info("Loading model components...")
        
        # Load model state dict
        logger.info("Loading model state...")
        model_state = torch.load("model_state.pt", map_location=torch.device('cpu'))
        
        # Load encoder data
        logger.info("Loading encoder data...")
        with open("encoder_data.json", "r") as f:
            encoder_data = json.load(f)
            
        # Load meta information
        with open("model_meta.json", "r") as f:
            meta_info = json.load(f)
            
        # Recreate encoder state
        encoder = EncoderStateLoader(
            num_bus=encoder_data["num_bus"],
            cat_bus=encoder_data["cat_bus"],
            bin_bus=encoder_data["bin_bus"],
            num_tpl=encoder_data["num_tpl"],
            cat_tpl=encoder_data["cat_tpl"],
            bin_tpl=encoder_data["bin_tpl"],
            states=encoder_data["states"],
            state_to_idx=encoder_data["state_to_idx"],
            vocab_bus=encoder_data["vocab_bus"],
            vocab_tpl=encoder_data["vocab_tpl"]
        )
        
        # Rebuild the model
        logger.info("Rebuilding model...")
        model = MatchingModel(encoder)
        model.load_state_dict(model_state)
        model.eval()
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model was saved at epoch: {meta_info['epoch']}")
        logger.info(f"Best validation AUC: {meta_info['best_auc']}")
        
        return model, encoder, meta_info
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    # Test loading
    model, encoder, meta_info = load_model()
    print("Model loaded successfully!")
    print(f"Model was saved at epoch: {meta_info['epoch']}")
    print(f"Best validation AUC: {meta_info['best_auc']}")
