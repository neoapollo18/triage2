from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np

# Initialize app
app = FastAPI()

# Load model checkpoint
checkpoint = torch.load("best_model.pt", map_location=torch.device('cpu'))

# Extract model and encoder
model_state = checkpoint["model_state_dict"]
encoder = checkpoint["encoder_state"]

# Rebuild the model (must match architecture)
from your_model_file import MatchingModel  # <- adjust this if your model is in a specific file

model = MatchingModel(encoder)
model.load_state_dict(model_state)
model.eval()

# --------------------------------
# Define request/response formats
# --------------------------------

class BusinessInput(BaseModel):
    business: dict
    threepl: dict

@app.post("/predict")
def predict_match(data: BusinessInput):
    try:
        # Parse business and 3PL input dictionaries
        bus_df = pd.DataFrame([data.business])
        tpl_df = pd.DataFrame([data.threepl])
        
        # Encode features
        bus_num, bus_cat, bus_bin = encoder.encode_business(bus_df)
        tpl_num, tpl_cat, tpl_bin = encoder.encode_3pl(tpl_df)
        
        # Reshape tensors
        bus_data = (bus_num, bus_cat, bus_bin)
        tpl_data = (tpl_num, tpl_cat, tpl_bin)
        
        with torch.no_grad():
            bus_emb, tpl_emb = model(bus_data, tpl_data)
            similarity = (torch.cosine_similarity(bus_emb, tpl_emb) + 1) / 2
            match_score = similarity.item()
        
        return {"match_score": match_score}
    
    except Exception as e:
        return {"error": str(e)}
