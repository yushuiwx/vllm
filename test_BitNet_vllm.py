# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_path = "/mnt/msranlp/xun/BieNet_finetuning/training_results/debug/inference/huggingface_ckpt"
sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=128)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "What is the capital of France?",
    "The capital of France is",
    "The capital of China is",
    "The capital of Japan is",
    "The capital of Germany is",
    "The capital of Italy is",
    "The capital of Spain is",
    "The capital of Canada is",
    "The capital of Australia is",
    "The capital of Brazil is",
    "The capital of Russia is",
]
def get_messages(prompt):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text
prompts = [
    get_messages(prompt)
    for prompt in prompts
]
# Create an LLM.
llm = LLM(model="/mnt/msranlp/xun/BieNet_finetuning/training_results/debug/inference/huggingface_ckpt", trust_remote_code=True)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
'''
Prompt: '<|begin_of_text|>Human: Hello, my name is\nAssistant:', Generated text: 'Hello! It seems like you might be looking for something specific. Could you please provide more details or specify what you need assistance with?'
Prompt: '<|begin_of_text|>Human: The president of the United States is\nAssistant:', Generated text: 'As of my last update in October 2023, the President of the United States is Joe Biden. He took office on January 20, 2021. Please verify this information with a current source to ensure it is up-to-date.'
Prompt: '<|begin_of_text|>Human: What is the capital of France?\nAssistant:', Generated text: 'The capital of France is Paris.'
Prompt: '<|begin_of_text|>Human: The capital of France is\nAssistant:', Generated text: 'The capital of France is Paris.'
Prompt: '<|begin_of_text|>Human: The capital of China is\nAssistant:', Generated text: 'The capital of China is Beijing.'
Prompt: '<|begin_of_text|>Human: The capital of Japan is\nAssistant:', Generated text: 'The capital of Japan is Tokyo.'
Prompt: '<|begin_of_text|>Human: The capital of Germany is\nAssistant:', Generated text: 'The capital of Germany is Berlin.'
Prompt: '<|begin_of_text|>Human: The capital of Italy is\nAssistant:', Generated text: 'The capital of Italy is Rome.'
Prompt: '<|begin_of_text|>Human: The capital of Spain is\nAssistant:', Generated text: 'The capital of Spain is Madrid.'
Prompt: '<|begin_of_text|>Human: The capital of Canada is\nAssistant:', Generated text: 'The capital of Canada is Ottawa.'
Prompt: '<|begin_of_text|>Human: The capital of Australia is\nAssistant:', Generated text: 'The capital of Australia is Canberra.'
Prompt: '<|begin_of_text|>Human: The capital of Brazil is\nAssistant:', Generated text: 'The capital of Brazil is Brasília. It was officially inaugurated on March 21, 1960.'
Prompt: '<|begin_of_text|>Human: The capital of Russia is\nAssistant:', Generated text: 'The capital of Russia is Moscow.'
'''