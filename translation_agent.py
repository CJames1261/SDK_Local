#%%
from agents import Agent, Runner, set_tracing_disabled
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
import asyncio
import nest_asyncio
nest_asyncio.apply()

#%%
# ── Local server config ───────────────────────────────────────────────────────
local_client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

# Tracing tries to reach OpenAI's servers — disable it for local use.
set_tracing_disabled(True)

#%%
def local_model() -> OpenAIChatCompletionsModel:
    """Return a chat-completions model backed by the local server."""
    return OpenAIChatCompletionsModel(
        model="Mistral-7B-Instruct-v0.1",
        openai_client=local_client,
    )


# ── Agents ────────────────────────────────────────────────────────────────────
spanish_agent = Agent(
    name="Spanish agent",
    instructions="You translate the user's message to Spanish",
    model=local_model(),
)

french_agent = Agent(
    name="French agent",
    instructions="You translate the user's message to French",
    model=local_model(),
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate."
        "If asked for multiple translations, you call the relevant tools."
    ),
    model=local_model(),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
    ],
)

#%%
# ── Run ───────────────────────────────────────────────────────────────────────
async def main():
    user_id    = input("Enter your name: ").strip() or "anonymous"
    user_input = input("What would you like to translate? ").strip()

    result = await Runner.run(orchestrator_agent, input=user_input)
    print(f"\n[{user_id}] {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())

# %%
