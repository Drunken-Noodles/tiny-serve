from tiny_serve.config import QwenConfig
from tiny_serve.model import (
    QwenForCausalLM,
    load_hf_weights,
)
from tiny_serve.tokenizer.tokenizer import Tokenizer

from transformers import AutoTokenizer
import torch

class Engine:
    def __init__(self):
        config = QwenConfig.from_hf_config("checkpoints/Qwen3-0.6B/config.json") 
        
        # load model
        model = QwenForCausalLM(config)
        load_hf_weights(model, "checkpoints/Qwen3-0.6B")
        model.eval()

        # init tokenizer
        tokenizer = AutoTokenizer.from_pretrained("checkpoints/Qwen3-0.6B", local_files_only=True)
        tokenizer = Tokenizer(tokenizer)

        while True:
            prompt = input(">> ")
            if prompt.strip().lower() == "exit":
                break

            input_ids = torch.tensor(tokenizer.encode(prompt)).reshape(1, -1)

            max_new_tokens = 10
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    logits = model(input_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=1)
            
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            output = tokenizer.decode(input_ids[0]) # type: ignore
            print("Output:\n", output, "\n", flush=True)