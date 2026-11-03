from tools.ui_scrape import *

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from peft import PeftModel
import csv

MPATH = "meta-llama/Llama-3.1-70B-Instruct"
DEVICE_MAP = "auto"
print("HELLO")

tokenizer = AutoTokenizer.from_pretrained(MPATH)
model_base = AutoModelForCausalLM.from_pretrained(
    MPATH,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=DEVICE_MAP,
)

model2 = PeftModel.from_pretrained(model_base, "llama3.1-70b_fn").merge_and_unload()
# model2 = model_base

now = datetime.now()

prompt1 = "what is the maximum temperature that will allow tomatoes to be cultivated?"
prompt2 = "What is the minimum temperature recorded in the last 3 days?"
prompt3= ""

query1 = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
""" + prompt1 + """
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
<TODAY> Wednesday, 13 November 2024 </TODAY> """
query2 = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
""" + prompt2 + """
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
<TODAY> Wednesday, 13 November 2024 </TODAY> """
print("start")
pipe = pipeline(task="text-generation", model=model2, tokenizer=tokenizer, max_length = 2000, truncation = True)
result1 = pipe(query1)
res1 = result1[0]['generated_text'].split("<|start_header_id|>assistant<|end_header_id|>")[-1]
print(res1)
result2 = pipe(query2)
res2 = result2[0]['generated_text'].split("<|start_header_id|>assistant<|end_header_id|>")[-1]
print(res2)
print("END======")
