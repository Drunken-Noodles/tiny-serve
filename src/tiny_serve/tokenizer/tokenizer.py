from transformers import PreTrainedTokenizerBase

class Tokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer
 
    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id
    
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str | list[str]:
        return self.tokenizer.decode(token_ids)