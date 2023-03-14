import pandas as pd
import os
os.environ['WANDB_DISABLED'] = 'true'
#--------------------------------------------------------
import datasets
from datasets import Dataset
import torch
torch.cuda.is_available()
#--------------------------------------------------------
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
#--------------------------------------------------------
from endogpt.Preprocessor import preprocess_real
from endogpt.Preprocessor import preprocess_synthetic
from endogpt.Classifier import train_test_validation
#--------------------------------------------------------
def model(string):
    df = pd.read_csv(string)#('data/real.csv')
    real = preprocess_real(df)
    real["text"] = real['General Practitioner'] + real['Endoscopist'] + real['Instrument'] + 'INDICATIONS FOR PROCEDURE:' + real['Indications'] + 'Extent of Exam:'+ real['Extent of Exam'] +'FINDINGS: '+ real['findings']
    ds = Dataset.from_pandas(df)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained("tombrooks248/EndoGPT")#.to('cuda')
    def tokenize(batch):
        return tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
    ds = ds.map(tokenize, num_proc=4, batched=True)
    ds = ds.remove_columns(["text"])
    tts_ds = ds.train_test_split(test_size=0.3)
    tts_ds
    block_size = 64
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    lm_datasets = tts_ds.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    tokenizer.decode(lm_datasets["train"][17]["input_ids"])

    training_args = TrainingArguments(
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        output_dir="models",
        report_to=None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],

    )
    #trainer.train()
    return trainer
