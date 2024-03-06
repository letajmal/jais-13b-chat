from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def download_model(model_path, model_name):
    """Download a Hugging Face model and tokenizer to the specified directory"""
    # Check if the directory already exists
    if not os.path.exists(model_path):
        # Create the directory
        os.makedirs(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Save the model and tokenizer to the specified directory
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

# For this demo, download the English-French and French-English models
download_model('models/', 'core42/jais-13b-chat')

# Load model directly
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained("core42/jais-13b-chat", trust_remote_code=True)