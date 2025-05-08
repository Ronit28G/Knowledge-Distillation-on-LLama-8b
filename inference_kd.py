from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import torch



model_path = "/scratch/rg4881/Project/kd_student_final_merged"



tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")



student_gen = pipeline(

    "text-generation",

    model=model,

    tokenizer=tokenizer,

    return_full_text=False,


)



prompt = "Classify the text:\n<text>\nIt was a beautiful day.\n</text>\nChoose from: positive, negative, neutral"

import time
start_time = time.time()

output = student_gen(prompt, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)

end_time = time.time()

inference_time = end_time - start_time

print(inference_time)

print("Prediction:", output[0]["generated_text"])


