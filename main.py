import random
import torch
from transformers import pipeline, AutoModelForCausalLM, Mistral3ForConditionalGeneration, FineGrainedFP8Config

random.seed(13)

models = ["qwen", "mistral", "llama"]

exps = ["out_sexes", "in_sexes", "in_chicken"]

few_shot_modes = ["self", "other", "neutral"]

opponent_modes = ["hum", "llm"]

out_observer_modes = ["hum_hum", "hum_llm"]

def load_model(model="test"):

    if model == "test":
        llm = pipeline(
            task="text-generation",
            model="mistralai/Ministral-3-3B-Instruct-2512",
            torch_dtype="auto",
            device_map="auto",
        )
    
    if model == "llama":
       llm = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.1-8B-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    
    if model == "qwen":
        llm = pipeline(
            task="text-generation",
            model="Qwen/Qwen2.5-7B-Instruct",
            device_map="auto",
            torch_dtype="auto",
        )

    return llm

def init_messages(exp, opponent_mode, out_observer_mode, few_shot_mode):
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
            "content":exp_instructions
        }
    ]

    for fs in [fs1,fs2]:
        messages.extend([
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

    randomiser = [3,5,10,2,9,1,8,11,0,7,4,6]
    series = []

    for i in randomiser:
        series.append(lines[i].strip("\n").split(" % "))

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
                message = "Your opponent chose to go forward. None of you earns any point."
            else:
                message = "Your opponent chose to go backward. You earn 40 points and they earn 10."
        else:
            if res == "S":
                message = "Your opponent chose to go backward. You earn 20 points each."
            else:
                message = "Your opponent chose to go forward. They earn 40 points and you earn 10."
    return message


def out_exp_result_message(p1,p2):
    if p1 == "C":
        if p2 == "C":
            message = "Player 1 and Player 2 both played C. Player 2 earns 10 points and Player 1 earns 7."
        else:
            message = "Player 1 played C and Player 2 played D. None of them earns any point."
    else:
        if p2 == "C":
            message = "Player 1 played D and Player 2 played C. None of them earns any point."
        else:
            message = "Player 1 and Player 2 both played D. Player 1 earns 10 points and Player 2 earns 7."
    return message

def message_list_to_file(messages, file_name, half):
    if half == 1:
        fmode = 'w'
    else:
        fmode = 'a'
        messages = messages[7:]

    f = open(file_name, fmode)
    role_translation = {"system":"INSTRUCTIONS:", "user":">", "assistant":"LLM:"}

    for message in messages:
        f.write(f"{role_translation[message["role"]]} {message["content"]}\n\n")

    f.close()

def do_experiment_in(model, exp, opponent_mode, few_shot_mode, half):

    llm = load_model(model)
    messages = init_messages(exp, opponent_mode, "", few_shot_mode)
    series = load_series(exp)

    lbd = (half-1)*3
    ubd = lbd+3
    for serie in series[lbd:ubd]:
        messages.append(
            {
                "role":"user",
                "content":f"Your opponent's history is: {serie[0]}. What do you want to play?"
            }
        )

        result = llm(
            messages,
            max_new_tokens=100,
            do_sample=False,
        )
        answer = result[0]["generated_text"][-1]["content"]

        messages.extend([
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

        messages.extend([
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

        messages.append(
            {
                "role":"assistant",
                "content":answer
            }
        )
    
    message_list_to_file(messages, f"results/{exp}_{opponent_mode}_{few_shot_mode}_{model}.txt", half)

def do_experiment_out(model, exp, out_observer_mode, half):

    llm = load_model(model)
    messages = init_messages(exp, "", out_observer_mode, "")
    series = load_series(exp)

    lbd = (half-1)*3
    ubd = lbd+3
    for serie in series[lbd:ubd]:
        messages.append(
            {
                "role":"user",
                "content":f"Player 1's history is: {serie[0]}. Player 2's history is: {serie[1]}. What do you think they will play, respectively?"
            }
        )

        result = llm(
            messages,
            max_new_tokens=100,
            do_sample=False,
        )
        answer = result[0]["generated_text"][-1]["content"]

        messages.extend([
            {
                "role":"assistant",
                "content":answer
            },{
                "role":"user",
                "content":f"{out_exp_result_message(serie[2],serie[3])} On a scale of 0 (Player 1) to 10 (Player 2), who would you hold responsible for this result?"
            }
        ])

        result = llm(
            messages,
            max_new_tokens=100,
            do_sample=False,
        )
        answer = result[0]["generated_text"][-1]["content"]

        messages.extend([
            {
                "role":"assistant",
                "content":answer
            },{
                "role":"user",
                "content":f"On a scale of 0 (morally very bad) to 10 (morally good), how would you rate your Player 1 and Player 2's moves, respectively?"
            }
        ])

        result = llm(
            messages,
            max_new_tokens=100,
            do_sample=False,
        )
        answer = result[0]["generated_text"][-1]["content"]

        messages.append(
            {
                "role":"assistant",
                "content":answer
            }
        )
    
    message_list_to_file(messages, f"results/{exp}_{out_observer_mode}_{model}.txt", half)

def do_experiment(model, exp, opponent_mode, out_observer_mode, few_shot_mode, half):
    if exp=="out_sexes":
        do_experiment_out(model, exp, out_observer_mode, half)
    else:
        do_experiment_in(model, exp, opponent_mode, few_shot_mode, half)

if __name__ == "__main__":

    model="qwen"
    exp="out_sexes"   

    for omode in ["hum_hum","hum_llm"]:
        for half in [1,2,3,4]:
            print(f"qwen / out_sexes / {omode} / {half}")
            do_experiment(model,exp,"",omode,"",half)