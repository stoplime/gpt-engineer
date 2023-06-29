from __future__ import annotations

import logging

import openai

logger = logging.getLogger(__name__)


class AI:
    def __init__(self, model="gpt-4", temperature=0.1, local_model=False, openai_base="http://localhost:8080/v1", openai_key="-"):
        self.model = model
        self.temperature = temperature
        self.local_model = local_model
        self.openai_base = openai_base
        self.openai_key = openai_key

        if not local_model:
            try:
                openai.Model.retrieve(model)
            except openai.InvalidRequestError:
                print(
                    f"Model {model} not available for provided API key. Reverting "
                    "to gpt-3.5-turbo. Sign up for the GPT-4 wait list here: "
                    "https://openai.com/waitlist/gpt-4-api"
                )
                self.model = "gpt-3.5-turbo"

    def start(self, system, user):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        return self.next(messages)

    def fsystem(self, msg):
        return {"role": "system", "content": msg}

    def fuser(self, msg):
        return {"role": "user", "content": msg}

    def fassistant(self, msg):
        return {"role": "assistant", "content": msg}

    def next(self, messages: list[dict[str, str]], prompt=None):
        if prompt:
            messages += [{"role": "user", "content": prompt}]

        logger.debug(f"Creating a new chat completion: {messages}")
        if self.local_model:
            response = openai.ChatCompletion.create(
                messages=messages,
                stream=True,
                model=self.model,
                temperature=self.temperature,
                api_base=self.openai_base,
                api_key=self.openai_key,
            )
            # print(list(response))
        else:
            response = openai.ChatCompletion.create(
                messages=messages,
                stream=True,
                model=self.model,
                temperature=self.temperature,
            )

        chat = []
        for chunk in response:
            delta = chunk["choices"][0].get("delta")
            if delta is None:
                continue
            msg = delta.get("content", "")
            print(msg, end="")
            chat.append(msg)
        print()
        messages += [{"role": "assistant", "content": "".join(chat)}]
        logger.debug(f"Chat completion finished: {messages}")
        return messages
