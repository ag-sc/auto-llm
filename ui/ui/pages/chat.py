import os
from openai import AsyncOpenAI
import reflex as rx

from ..state.user import User
from ..templates import template
from .. import styles

class MockAsyncIterator:
    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.items):
            result = self.items[self.index]
            self.index += 1
            return result
        else:
            raise StopAsyncIteration

# 1. Your dummy data (using the SimpleNamespace trick from before)
from types import SimpleNamespace
dummy_chunks = [
    SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))]),
    SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=" world"))])
]





class State(rx.State):
    question: str
    chat_history: list[tuple[str, str]] = []

    async def answer(self):
        # client = AsyncOpenAI(
        #     api_key=os.environ["OPENAI_API_KEY"]
        # )

        # Start streaming completion from OpenAI
        # session = await client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {"role": "user", "content": self.question}
        #     ],
        #     temperature=0.7,
        #     stream=True,
        # )

    #     session = {
    #     "id": "chatcmpl-123",
    #     "object": "chat.completion.chunk",
    #     "created": 1700000000,
    #     "model": "gpt-4o-mini",
    #     "choices": [
    #         {
    #             "index": 0,
    #             "delta": {"content": "Hello"}, # This changes per chunk
    #             "logprobs": None,
    #             "finish_reason": None      # Only the last chunk has a finish_reason
    #         }
    #     ]
    # }

        # 2. Wrap it so 'async for' works
        session = MockAsyncIterator(dummy_chunks)

        # Initialize response and update UI
        answer = ""
        self.chat_history.append((self.question, answer))
        self.question = ""
        yield

        # Process streaming response
        async for item in session:
            if hasattr(item.choices[0].delta, "content"):
                if item.choices[0].delta.content is None:
                    break
                answer += item.choices[0].delta.content
                self.chat_history[-1] = (
                    self.chat_history[-1][0],
                    answer,
                )
                yield

def qa(question: str, answer: str) -> rx.Component:
    return rx.box(
        rx.box(
            rx.text(question, style=styles.question_style),
            text_align="right",
        ),
        rx.box(
            rx.text(answer, style=styles.answer_style),
            text_align="left",
        ),
        margin_y="1em",
        width="100%",
    )


def chat() -> rx.Component:
    return rx.card(
        rx.scroll_area(
            rx.vstack(
                rx.foreach(
                    State.chat_history,
                    lambda messages: qa(messages[0], messages[1]),
                ),
                spacing="2",
                width="100%",
            ),
            height="60vh",
            width="100%",
        ),
        width="60%",
    )


def action_bar() -> rx.Component:
    return rx.hstack(
        rx.input(
            value=State.question,
            placeholder="Ask a question",
            on_change=State.set_question,
            style=styles.input_style,
            variant="soft"
        ),
        rx.button(
            "Ask",
            on_click=State.answer,
            style=styles.button_style,
        ),
        width="60%",
    )


def settings():
    return rx.card(
        rx.hstack(
            rx.hstack(
                rx.icon("hash", size=20),
                rx.text("Port", weight="bold"),
                rx.text_field(placeholder="Enter the port", name="port"),
                align="center",
            ),
            rx.spacer(),
            rx.hstack(
                rx.icon("chip", size=20),
                rx.text("Model", weight="bold"),
                rx.select(["model 1", "model 2", "model 3"], placeholder="Select a model", name="model"),
                align="center",
            ),
        ),
        width="60%",
    )

@template(route="/chat", title="Chat", on_load=User.check_logged_in)
def index() -> rx.Component:
    return rx.center(
        rx.vstack(
            chat(),
            action_bar(),
            settings(),
            align="center",
            width="100%",
        ),
        width="100%",
        height="70vh",
    )
