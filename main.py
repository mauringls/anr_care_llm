import torch
from transformers import pipeline, AutoModelForCausalLM, Mistral3ForConditionalGeneration, FineGrainedFP8Config

qwen = "Qwen/Qwen2.5-7B-Instruct"
mistral = "mistralai/Mistral-7B-Instruct-v0.3"
llama = "meta-llama/Llama-3.1-8B-Instruct"

models = [qwen, mistral, llama]
models_names = ["qwen", "mistral", "llama"]

exps = ["out_sexes", "in_sexes", "in_chicken"]

few_shot_modes = ["self", "other", "neutral"]

opponent_modes = ["hum", "llm"]

out_observer_modes = ["hum_hum", "hum_llm"]

def load_model(model=""):
    llm = pipeline(
        task="text-generation",
        model="mistralai/Ministral-3-3B-Instruct-2512",
        torch_dtype="auto",
        device_map="auto",
    )
    return llm

def init_messages(exp, few_shot_mode, opponent_mode, out_observer_mode):
    return []

def load_series(exp):
    return []


def do_experiment_in(
    model="llama",
    exp="out_sexes",
    few_shot_mode="neutral",
    opponent_mode="hum",
    out_observer_mode="hum_hum"):

    messages = init_messages(exp, few_shot_mode, opponent_mode, out_observer_mode)
    series = load_series(exp)





if __name__ == "__main__":

    messages = [
        {
            "role": "user",
            "content": "Write a Python code sample to print Hello World"
        }
    ]

    llm = load_model()

    result = llm(
        messages,
        max_new_tokens=100,
        do_sample=False,
    )

    print(result[0]["generated_text"][-1]["content"])