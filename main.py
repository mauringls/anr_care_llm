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

def init_messages(exp, opponent_mode, out_observer_mode, few_shot_mode):
    with open("instructions/general_intructions.txt", 'r') as f:
        general_instructions = f.read()

    if exp == "out_sexes":
        with open(f"instructions/exp_specific/{exp}_{out_observer_mode}_instruct.txt", 'r') as f:
            exp_instructions = f.read()

        with open(f"instructions/few_shots/{exp}_fs.txt",'r') as f:
            lines = f.readlines()
            fs1 = lines[0].split(" % ")
            fs2 = lines[1].split(" % ")
    
    else:
        with open(f"instructions/exp_specific/{exp}_{opponent_mode}_instruct.txt", 'r') as f:
            exp_instructions = f.read()

        with open(f"instructions/few_shots/{exp}_{few_shot_mode}_fs.txt",'r') as f:
            lines = f.readlines()
            fs1 = lines[0].split(" % ")
            fs2 = lines[1].split(" % ")

    messages = [
        {
            "role":"system",
            "content":general_instructions
        },{
            "role":"system",
            "content":exp_instructions
        }
    ]

    for fs in [fs1,fs2]:
        messages.append([
            {
                "role":"user",
                "content":fs[0]
            },{
                "role":"assistant",
                "content":fs[1]
            },{
                "role":"user",
                "content":fs[2]
            },{
                "role":"assistant",
                "content":fs[3]
            },{
                "role":"user",
                "content":fs[4]
            },{
                "role":"assistant",
                "content":fs[5]
            }
        ])


    return messages

def load_series(exp):
    return []


def do_experiment_in(
    model="llama",
    exp="out_sexes",
    opponent_mode="hum",
    out_observer_mode="hum_hum",
    few_shot_mode="neutral"):

    messages = init_messages(exp, opponent_mode, out_observer_mode, few_shot_mode,)
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