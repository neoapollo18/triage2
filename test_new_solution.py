import torch
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the model classes
from _3pl_matching_model import MatchingModel

# This is a simplified version of the EncoderState class that we'll use for deployment
class EncoderState:
    """A deployment-friendly version of FeatureEncoder that works with pickle across modules"""
    
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
        self.vocab_bus = vocab_bus
        self.vocab_tpl = vocab_tpl

def test_alternative_loading():
    """Test our alternative loading strategy that avoids pickle module issues"""
    try:
        logger.info("Testing alternative model loading approach")
        
        # Step 1: Load the original model checkpoint
        try:
            # Try the PyTorch 2.6+ approach
            logger.info("Attempting to load with PyTorch 2.6+ approach")
            checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'), weights_only=False)
        except TypeError:
            # Fall back to older PyTorch versions
            logger.info("Falling back to standard PyTorch loading")
            checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))
        
        logger.info("Successfully loaded original model checkpoint")
        
        # Step 2: Extract components we need
        model_state = checkpoint["model_state_dict"]
        original_encoder = checkpoint["encoder_state"]
        
        # Step 3: Create deployment-friendly encoder
        logger.info("Creating deployment-friendly encoder")
        encoder = EncoderState(
            num_bus=original_encoder.num_bus,
            cat_bus=original_encoder.cat_bus,
            bin_bus=original_encoder.bin_bus,
            num_tpl=original_encoder.num_tpl,
            cat_tpl=original_encoder.cat_tpl,
            bin_tpl=original_encoder.bin_tpl,
            states=original_encoder.states,
            state_to_idx=original_encoder.state_to_idx,
            vocab_bus=original_encoder.vocab_bus,
            vocab_tpl=original_encoder.vocab_tpl
        )
        
        # Step 4: Initialize model with new encoder
        logger.info("Initializing model with deployment-friendly encoder")
        model = MatchingModel(encoder)
        
        # Step 5: Load state dictionary
        logger.info("Loading model weights")
        model.load_state_dict(model_state)
        model.eval()
        
        # Success!
        logger.info(f"Model was saved at epoch: {checkpoint.get('epoch')}")
        logger.info(f"Best validation AUC: {checkpoint.get('best_auc')}")
        
        print("\n✅ SUCCESS: Alternative loading strategy works!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing alternative loading: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print("\n❌ FAILURE: Alternative loading strategy failed!")
        return False

if __name__ == "__main__":
    success = test_alternative_loading()
    sys.exit(0 if success else 1)
