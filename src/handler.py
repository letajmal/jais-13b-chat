from transformers import pipeline
import runpod

pipe = pipeline("text-generation", model="core42/jais-13b-chat")


def handler(job):
    job_input = job['input']
    message = job_input.get('message', 'Hi!')
    return pipe(message)


runpod.serverless.start({"handler": handler})