from tiny_serve.tokenizer.tokenizer import Tokenizer

from transformers import AutoTokenizer

def test_tokenizer() -> None:
    tokenizer = AutoTokenizer.from_pretrained("checkpoints/Qwen3-0.6B", local_files_only=True)
    tokenizer = Tokenizer(tokenizer)
    text = "Hello, how are you?"
    encoded_text = tokenizer.encode(text)
    decoded_text = tokenizer.decode(encoded_text)
    assert decoded_text == text