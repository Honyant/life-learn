"""Shared inference module.

Provides two classes with the same ``.generate()`` interface:

* ``LocalInference``  – runs a HuggingFace model on GPU (for chat responses)
* ``OpenAIInference`` – calls the OpenAI API (for structured tasks like
  correction detection and augmentation where small local models fail)
"""
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalInference:
    """Manages a locally loaded model for chat inference."""

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
                repetition_penalty=1.3,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def unload(self):
        """Free GPU memory. Call before loading training models."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


class OpenAIInference:
    """Calls the OpenAI API with the same ``.generate()`` signature as LocalInference.

    Use this for structured tasks (correction detection, augmentation) where
    small local models produce unreliable JSON output.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI()

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """Generate a completion via the OpenAI API.

        ``do_sample`` is accepted for interface compatibility but ignored
        (temperature=0 is used for deterministic output instead).
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
        )
        return response.choices[0].message.content

    def unload(self):
        """No-op — included so OpenAIInference is a drop-in for LocalInference."""
        pass
