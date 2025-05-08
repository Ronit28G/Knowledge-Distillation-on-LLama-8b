import re

import time

import torch

import pandas as pd

import torch.cuda.nvtx as nvtx

from transformers import (

    pipeline,

    AutoTokenizer,

    AutoModelForCausalLM,

)



# ---------- Configuration ----------

model_id = "./Llama-3.1-8B-Instruct"

class_names = ["positive", "negative", "neutral"]

input_json = "test_data.json"



# ---------- Load Test Data ----------

test_df = pd.read_json(input_json)



def create_dataset(df):

    return [{"input": row["input"], "output": row["output"]} for _, row in df.iterrows()]



test_rows = create_dataset(test_df)



# ---------- Prompt Construction ----------

def create_prompt(statement: str, class_names: list):

    prompt = """

Classify the text for one of the categories:



<text>

{text}

</text>



Choose from one of the category:

{classes}

Only choose one category, the most appropriate one. Reply only with the category.

""".strip()

    return prompt.format(text=statement, classes=", ".join(class_names))



# ---------- Load Tokenizer and Model ----------

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(

    model_id,

    torch_dtype=torch.bfloat16,

).to("cuda")



# ---------- Pipeline ----------

teacher_gen = pipeline(

    "text-generation",

    model=model,

    tokenizer=tokenizer,

    return_full_text=False,

    return_dict_in_generate=False,

    output_scores=False,

    device=0

)



# ---------- Run a Single Prompt ----------

row = test_rows[0]



nvtx.range_push("Prompt Construction")

prompt = create_prompt(row["input"], class_names)

nvtx.range_pop()



nvtx.range_push("Model Inference")

start_time = time.time()

outputs = teacher_gen(

    prompt,

    max_new_tokens=32,

    pad_token_id=tokenizer.eos_token_id

)

end_time = time.time()

inference_time = end_time - start_time
print(inference_time)

nvtx.range_pop()



# ---------- Clean & Save Result ----------

nvtx.range_push("Postprocessing & Save")

regex = r"^\W+|\W+$"

generated = outputs[0]["generated_text"]

cleaned_pred = re.sub(regex, "", generated.lower())



eval_df = pd.DataFrame({

    "text": [row["input"]],

    "label": [row["output"]],

    "llama_8b output": [cleaned_pred]

})



eval_df.to_csv("inference_result_single.csv", index=False)

nvtx.range_pop()
