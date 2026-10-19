import random
import torch
from transformers import pipeline, AutoModelForCausalLM, Mistral3ForConditionalGeneration, FineGrainedFP8Config

random.seed(13)

MAX_SERIES = 10

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
            fs1 = lines[0].strip("\n").split(" % ")
            fs2 = lines[1].strip("\n").split(" % ")
    
    else:
        with open(f"instructions/exp_specific/{exp}_{opponent_mode}_instruct.txt", 'r') as f:
            exp_instructions = f.read()

        with open(f"instructions/few_shots/{exp}_{few_shot_mode}_fs.txt",'r') as f:
            lines = f.readlines()
            fs1 = lines[0].strip("\n").split(" % ")
            fs2 = lines[1].strip("\n").split(" % ")

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
    with open(f"instructions/series/{exp}_series.txt", 'r') as f:
            lines = f.readlines()

    series = []

    for line in random.shuffle(lines):
        series.append(line.strip("\n").split(" % "))

    return series

def in_exp_result_message(exp,ans,res):
    if exp == "in_sexes":
        if ans == "C":
            if res == "S":
                message = "Your opponent played C. They earn 10 points and you earn 7."
            else:
                message = "Your opponent played D. You earn 10 points and they earn 7."
        else:
            if res == "S":
                message = "Your opponent played D. None of you earns any point."
            else:
                message = "Your opponent played C. None of you earns any point."
    else:
        if ans == "F":
            if res == "S":
                message = "Your opponent chose to go forwards. None of you earns any point."
            else:
                message = "Your opponent chose to go backwards. You earn 40 points and they earn 10."
        else:
            if res == "S":
                message = "Your opponent chose to go backards. You earn 20 points each."
            else:
                message = "Your opponent chose to go forwards. They earn 40 points and you earn 10."

def message_list_to_file(messages, file_name):
    f = open(file_name, 'w')
    role_translation = {"system":"INSTRUCTIONS:", "user":">", "assistant":"LLM:"}

    for message in messages:
        f.write(f"{role_translation[message["role"]]} {message["content"]}\n\n")

    f.close()

def do_experiment_in(
    model="llama",
    exp="out_sexes",
    opponent_mode="hum",
    few_shot_mode="neutral"):

    messages = init_messages(exp, opponent_mode, "", few_shot_mode,)
    series = load_series(exp)

    for serie in series[:MAX_SERIES]:
        messages.append([
            {
                "role":"user",
                "content":f"Your opponent's history is: {serie[0]}. What do you want to play?"
            }
        ])

        result = llm(
            messages,
            max_new_tokens=100,
            do_sample=False,
        )
        answer = result[0]["generated_text"][-1]["content"]

        messages.append([
            {
                "role":"assistant",
                "content":answer
            },{
                "role":"user",
                "content":f"{in_exp_result_message(exp,answer,serie[1])} On a scale of 0 (you) to 10 (your opponent), who would you hold responsible for this result?"
            }
        ])

        result = llm(
            messages,
            max_new_tokens=100,
            do_sample=False,
        )
        answer = result[0]["generated_text"][-1]["content"]

        messages.append([
            {
                "role":"assistant",
                "content":answer
            },{
                "role":"user",
                "content":f"On a scale of 0 (morally very bad) to 10 (morally good), how would you rate your opponent's move?"
            }
        ])

        result = llm(
            messages,
            max_new_tokens=100,
            do_sample=False,
        )
        answer = result[0]["generated_text"][-1]["content"]

        messages.append([
            {
                "role":"assistant",
                "content":answer
            }
        ])
    
    message_list_to_file(messages, f"results/{exp}_{opponent_mode}_{few_shot_mode}_{model}.txt")






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