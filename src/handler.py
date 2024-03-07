# from transformers import AutoTokenizer, AutoModelForCausalLM
# import runpod

# pipe = pipeline("text-generation", model="core42/jais-13b-chat", trust_remote_code=True)

# def handler(job):
#     job_input = job['input']
#     message = job_input.get('message', 'Hi!')
#     return pipe(message)


# runpod.serverless.start({"handler": handler})

import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer from the saved directory
tokenizer = AutoTokenizer.from_pretrained("models/")
model = AutoModelForCausalLM.from_pretrained("models/", trust_remote_code=True)

def handler(job):
    """Handler function that uses the loaded model for inference."""
    job_input = job['input']

    name = job_input.get('message', 'Hi!')

    # Generate a response using the model
    with tokenizer.as_target_tokenizer():  # Use target tokenizer for conversation model
        input_ids = tokenizer.encode(message, return_tensors="pt")
        output = model.generate(input_ids, max_length=50)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

runpod.serverless.start({"handler": handler})
