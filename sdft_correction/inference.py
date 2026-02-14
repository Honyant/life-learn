"""Shared inference module using a local HuggingFace model."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalInference:
    """Manages a locally loaded model for inference (correction detection, augmentation, chat)."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """Generate a completion from a list of chat messages.

        Returns the assistant response text only.
        """
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def unload(self):
        """Free GPU memory. Call before loading training models."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
