"""
llm_client.py
Pluggable "generation" step, so the rest of the pipeline (router, retrieval,
prompt building) doesn't care which LLM backend is used.

Two backends:
  - ExtractiveLLM (default): no API key needed, runs fully offline. It
    doesn't call any model -- it deterministically formats the retrieved
    context into a readable answer. This keeps the whole project runnable
    end-to-end without any paid API, while still exercising the full
    routing + retrieval + prompting pipeline.
  - OpenAILLM (optional): calls a real chat model via the OpenAI SDK, used
    only if `--use-openai` is passed AND an OPENAI_API_KEY environment
    variable is set. Falls back to ExtractiveLLM with a warning if the SDK
    or key is missing.
"""

import os
from abc import ABC, abstractmethod

from src.config import Config


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


class ExtractiveLLM(BaseLLM):
    """
    Deterministic, offline 'generation': just cleans up and returns the
    context in a readable way. This is intentionally simple -- swap in
    OpenAILLM (or any other BaseLLM implementation) for real free-text
    generation without touching any other module.
    """

    _CONTEXT_HEADERS = ("Database records:", "Database record:", "Relevant policy/FAQ excerpts:", "Policy/FAQ excerpts:")
    _EMPTY_MARKERS = ("No matching records", "No relevant documents")

    def generate(self, prompt: str) -> str:
        # Pull out the context block(s) the prompt already assembled and
        # present them as a direct answer rather than re-generating text.
        marker = "User question:"
        if marker in prompt:
            context_part = prompt.split(marker)[0]
        else:
            context_part = prompt
        context_part = context_part.strip()

        if not context_part:
            return "I couldn't find relevant information to answer this question."

        # A hybrid prompt can have ONE source come up empty while the other
        # has real content (e.g. no matching SQL record, but a relevant
        # policy doc). Check specifically the text following each known
        # context header, ignoring the instruction preamble entirely --
        # otherwise the preamble's own prose gets mistaken for "content".
        found_real_content = False
        for header in self._CONTEXT_HEADERS:
            if header not in context_part:
                continue
            section = context_part.split(header, 1)[1]
            # Section runs until the next header or end of context_part.
            for other_header in self._CONTEXT_HEADERS:
                if other_header != header and other_header in section:
                    section = section.split(other_header, 1)[0]
            section = section.strip()
            if section and not any(m in section for m in self._EMPTY_MARKERS):
                found_real_content = True
                break

        if not found_real_content:
            return "I couldn't find relevant information to answer this question."

        return (
            "Based on the retrieved information:\n\n" + context_part +
            "\n\n(This is an extractive summary of retrieved context. "
            "Set --use-openai with OPENAI_API_KEY to get a fully generated "
            "natural-language answer instead.)"
        )


class OpenAILLM(BaseLLM):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        try:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self.client = OpenAI(api_key=api_key)
            self.available = True
        except Exception as e:
            self.available = False
            self._error = str(e)

    def generate(self, prompt: str) -> str:
        if not self.available:
            fallback = ExtractiveLLM()
            return (f"[OpenAI unavailable: {self._error}. Falling back to extractive mode]\n\n"
                    + fallback.generate(prompt))

        response = self.client.chat.completions.create(
            model=self.cfg.llm.openai_model,
            temperature=self.cfg.llm.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


def build_llm(cfg: Config) -> BaseLLM:
    if cfg.llm.provider == "openai":
        return OpenAILLM(cfg)
    return ExtractiveLLM()
