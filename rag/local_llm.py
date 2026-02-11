from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

SYSTEM_PROMPT = """You are a Kubernetes documentation assistant.

CRITICAL RULES:
1. Answer ONLY using information from the provided context
2. Do NOT use your general Kubernetes knowledge
3. If the context doesn't contain the answer, ONLY output: "I don't know"
4. Be specific and cite relevant details from the context
5. Keep answers clear and concise"""

class LocalLLM:
    def __init__(self, model_name: str, device: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            do_sample=False,
            return_full_text=False,
        )

        self.device = device

    def generate_chat(
        self, user_message: str, max_new_tokens: int = 512, system_message: str = None
    ) -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        out = self.pipe(formatted_prompt, max_new_tokens=max_new_tokens)[0][
            "generated_text"
        ]

        return out.replace("<|im_end|>", "").strip()

    def generate_answer(self, user_message: str, max_new_tokens: int = 512) -> str:
        """Generate answer with RAG system prompt"""
        return self.generate_chat(
            user_message=user_message,
            max_new_tokens=max_new_tokens,
            system_message=SYSTEM_PROMPT,
        )

    def generate_judgment(self, user_message: str) -> str:
        """Generate YES/NO judgment with strict system prompt"""
        system_prompt = """You are an evaluation assistant. Your job is to judge answers based on given criteria.

CRITICAL RULES:
1. You must respond with EXACTLY one word: YES or NO
2. Do not add explanations, reasoning, or any other text
3. Do not add punctuation
4. Just output: YES or NO"""

        verdict = self.generate_chat(
            user_message=user_message, max_new_tokens=5, system_message=system_prompt
        )

        # Parse to ensure YES or NO
        verdict = verdict.strip().upper()
        first_word = verdict.split()[0] if verdict else ""

        if "YES" in first_word:
            return "YES"
        elif "NO" in first_word:
            return "NO"
        else:
            print(f"Unclear judgment: '{verdict}' -> defaulting to NO")
            return "NO"
