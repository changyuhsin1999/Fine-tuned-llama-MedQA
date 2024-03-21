import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from huggingface_hub import notebook_login
from random import randrange
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

def load_model_and_tokenizer(model_name):
    use_flash_attention = False

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        use_cache=False,
        use_flash_attention_2=use_flash_attention,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer


class DatasetLoader:
    def __init__(self, dataset_name):
        self.dataset = load_dataset(dataset_name)

    def train_test_split(self, test_size=0.2):
        train_test_split = self.dataset["train"].train_test_split(test_size=test_size)
        self.dataset = DatasetDict({
            'train': train_test_split['train'],
            'test': train_test_split['test']
        })

def format_prompt(sample):
    return f"""
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    please give the response in a simple format of A: answer and do not give any explanation. please do not add special characters or extra spaces in the response.

    ### Instruction:
    {sample["instruction"]}

    ### Input:
    {sample["input"]}

    ### Response:

    """

class ModelTrainer:
    def __init__(self, model_name, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def prepare_for_training(self, dataset):
        peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.1,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, peft_config)
        
        args = TrainingArguments(
            output_dir="finetuned-llama-7b-chat-hf-with-medQA",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=10,
            save_strategy="epoch",
            learning_rate=2e-4,
            fp16=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            disable_tqdm=False
        )
        
        max_seq_length = 1024
        
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset["train"],
            peft_config=peft_config,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            packing=True,
            formatting_func=format_prompt,
            args=args,
        )

    def train(self):
        self.trainer.train()

    def evaluate(self, dataset):
        correct_predictions = 0
        total_predictions = 0
        num_samples = 100

        for _ in range(num_samples):
            sample = dataset[randrange(len(dataset))]
            prompt = format_prompt(sample)
            input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
            outputs = self.model.generate(input_ids=input_ids, max_new_tokens=512, do_sample=True, top_p=0.6, temperature=0.9)
            generated_response = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
            normalized_generated_response = generated_response.strip().lower()
            normalized_ground_truth = sample['output'].strip().lower()

            if normalized_generated_response == normalized_ground_truth:
                correct_predictions += 1
            total_predictions += 1

        accuracy = correct_predictions / total_predictions
        print(f"Accuracy: {accuracy*100:.2f}%")

def main():
    dataset_loader = DatasetLoader('medalpaca/medical_meadow_medqa')
    dataset_loader.train_test_split()
    model_trainer = ModelTrainer('meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-7b-chat-hf')
    model_trainer.prepare_for_training(dataset_loader.dataset)
    model_trainer.train()
    model_trainer.evaluate(dataset_loader.dataset['test'])

if __name__ == '__main__':
    main()
