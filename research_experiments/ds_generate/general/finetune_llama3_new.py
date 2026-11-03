import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import (SFTTrainer, SFTConfig)

DEVICE_MAP = "auto"


df0_init = pd.read_csv('../meteo/dataset/STEP1_DS.csv').dropna()
df0 = pd.DataFrame()
# take only half of the dataset
df0_init = df0_init.sample(frac=1).reset_index(drop=True)
df0['question'] = df0_init['text']
df0['answer'] = "<TODAY> " + df0_init['ref_date'] + " </TODAY> <METEO> " + df0_init['label'] + " -->  </METEO> "

df1 = pd.read_csv('annotate/clean_annotated/STEP2_DS.csv').dropna()
df2 = pd.read_csv('annotate/clean_annotated/STEP3_DS.csv').dropna()
df5 = pd.read_csv('annotate/clean_annotated/meteomanz.csv').dropna()
df6 = pd.read_csv('annotate/clean_annotated/meteoblue.csv').dropna()
df7 = pd.read_csv('annotate/clean_annotated/meteoblue_generated.csv').dropna()
df9 = pd.read_csv('annotate/clean_annotated/annotated_seeds_version1.csv').dropna()
df10 = pd.read_csv('annotate/clean_annotated/annotated_seeds_version2.csv').dropna()
df11 = pd.read_csv('annotate/clean_annotated/annotated_meteo&op_minmaxavg.csv').dropna()

df12 = pd.read_csv('../meteo/dataset/TRAIN_DATASET.csv').dropna() #meteo
df12['question'] = df12['Input']
df12['answer'] = df12['Output']

df = pd.concat([df1, df2, df5, df9, df10, df11, df7, df6, df12], ignore_index=True)
# df = pd.concat([df1, df2, df5, df6, df7], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

df = pd.concat([df0, df], ignore_index=True)

llama_df_mine = pd.DataFrame()
llama_df_mine['text'] = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
<|eot_id|>
<|start_header_id|>user<|end_header_id|> """ + df['question'] + \
"""<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
""" + df['answer'] + "<|end_of_text|>"
llama_df = llama_df_mine
llama_df = llama_df.sample(frac=1).reset_index(drop=True) # shuffle the dataset
llama_df.to_csv("general_dataset.csv", index=False)

MPATH = "meta-llama/Llama-3.1-8B-Instruct"
dataset = Dataset.from_pandas(llama_df)
print(dataset)

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # stores weights in 4-bits
    bnb_4bit_quant_type="nf4", # quantization of floats to 4-bit integers
    bnb_4bit_compute_dtype=torch.float16, # compute in half precision
    bnb_4bit_use_double_quant=True # double quantization
)

model = AutoModelForCausalLM.from_pretrained(
    MPATH,
    quantization_config=bnb_config,
    # attn_implementation="flash_attention_2",
    device_map=DEVICE_MAP,
)
model.config.use_cache = True
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MPATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

training_arguments = TrainingArguments(
    output_dir="./results-8b",
    #max_steps=100,
    num_train_epochs=2,
    per_device_train_batch_size=1, # per GPU
    gradient_accumulation_steps=1, # Number of update steps to accumulate the gradients for
    optim="adamw_torch", # Optimizer to use
    save_steps=1545,# Save checkpoint every X updates steps
    logging_steps=20, # Log every X updates steps
    learning_rate=1.0e-04,
    lr_scheduler_type="cosine",
    weight_decay=0.1,
    fp16=False, # Half precision floats
    bf16=True, # set bf16 to True with an A100
    max_grad_norm=0.3, # gardiet clipping: If a gradient exceeds some threshold value, we clip that gradient to the threshold
    warmup_ratio=0, # Percentage of updates steps to warmup the learning rate over
    group_by_length=True, # Group sequences into batches with same length, saves memory and speeds up training considerably
    report_to="tensorboard" # TensorBoard provides tooling for tracking and visualizing metrics as well as visualizing models
)



trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True,
)

trainer.train()

trainer.model.save_pretrained("llama3-8b_fn")
