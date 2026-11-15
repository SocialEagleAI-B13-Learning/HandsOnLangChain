"""A tiny LangChain app traced by LangSmith — with no tracing code.

All tracing is switched on via environment variables (see .env):
    OPENAI_API_KEY     - your OpenAI key
    LANGSMITH_TRACING  - "true" to enable tracing
    LANGSMITH_API_KEY  - your LangSmith key
    LANGSMITH_PROJECT  - project name so runs don't land in "default"

Run:  python traced_app.py
"""

import os  # standard library: lets us read environment variables with os.getenv()

from dotenv import load_dotenv  # reads key=value pairs from the .env file into the process environment
from langchain_core.output_parsers import StrOutputParser  # turns the model's message object into a plain string
from langchain_core.prompts import ChatPromptTemplate  # builds a reusable chat prompt with a {question} placeholder
from langchain_openai import ChatOpenAI  # LangChain's wrapper around OpenAI chat models (NOT the raw OpenAI SDK)

# Load .env BEFORE any LangChain objects are created.
# This is the only "setup" step — the LANGSMITH_* variables it loads are what
# switch tracing on. There is no tracing code anywhere in this file.
load_dotenv()


def verify_tracing_is_on() -> None:
    """Tracing needs no code — which means it can be silently off.

    Check the env vars before sending anything, so we don't go hunting
    the dashboard for runs that were never sent.
    """
    # Read the three env vars that control tracing. Defaults ("" / "default")
    # mirror what LangChain itself assumes when a variable is missing.
    tracing = os.getenv("LANGSMITH_TRACING", "")   # must be the string "true" for tracing to be on
    api_key = os.getenv("LANGSMITH_API_KEY", "")   # needed to authenticate when uploading traces
    project = os.getenv("LANGSMITH_PROJECT", "default")  # where runs appear in the LangSmith UI

    print("--- Tracing pre-flight check ---")
    print(f"LANGSMITH_TRACING = {tracing!r}")
    # Only show the first 12 chars of the key — enough to recognise it, safe to print.
    print(f"LANGSMITH_API_KEY = {'set (' + api_key[:12] + '...)' if api_key else 'NOT SET'}")
    print(f"LANGSMITH_PROJECT = {project!r}")

    # Three possible outcomes, checked in order of severity:
    if tracing.lower() != "true":
        # Tracing flag missing or not "true" → LangChain won't even try to trace.
        print("WARNING: tracing is OFF — runs will NOT appear in LangSmith.")
    elif not api_key:
        # Flag is on but there is no key → traces are attempted but rejected.
        print("WARNING: no LangSmith API key — traces cannot be sent.")
    else:
        # Both present → every chain.invoke() below will be recorded.
        print(f"Tracing is ON. Runs will land in project '{project}'.")
    print()


def main() -> None:
    # Fail fast: confirm tracing is configured before spending any OpenAI tokens.
    verify_tracing_is_on()

    # The chat model. OPENAI_API_KEY is picked up automatically from the environment.
    # temperature=0 → deterministic answers, easier to compare runs in LangSmith.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # A prompt template with two messages:
    #   system → fixed instructions sent with every request (keeps answers short/cheap)
    #   human  → the {question} placeholder filled in at invoke() time
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise assistant. Answer in at most 3 sentences."),
            ("human", "{question}"),
        ]
    )

    # LCEL pipe syntax: output of each step feeds the next.
    #   prompt (fills template) → llm (calls OpenAI) → parser (AIMessage → str)
    # Each invoke() of this chain shows up in LangSmith as ONE run with three
    # child steps, each with its own latency and token counts.
    chain = prompt | llm | StrOutputParser()

    # Three different prompts sent in one program run (the assignment requirement).
    questions = [
        "What is LangSmith and why would I use it with LangChain?",
        "Explain what a token is in the context of an LLM.",
        # Deliberately heavier prompt: asking for ~200 words forces many more
        # OUTPUT tokens → visibly higher latency & cost in the LangSmith dashboard.
        "Write a short story (around 200 words) about a debugging session "
        "where the villain turns out to be a missing environment variable.",
    ]

    # Send each prompt through the chain and print the answer.
    for i, question in enumerate(questions, start=1):  # start=1 → human-friendly numbering
        # invoke() fills {question}, calls OpenAI, parses the reply to a string.
        # Because LANGSMITH_TRACING=true, this call is automatically traced.
        answer = chain.invoke({"question": question})
        print(f"--- Prompt {i} ---")
        print(f"Q: {question}")
        print(f"A: {answer}")
        print()

    # Remind where to look: the project name comes from LANGSMITH_PROJECT in .env.
    print("Done. Open https://smith.langchain.com and check the project "
          f"'{os.getenv('LANGSMITH_PROJECT', 'default')}' for 3 traced runs.")


# Standard Python entry-point guard: run main() only when the file is executed
# directly (python traced_app.py), not when it is imported as a module.
if __name__ == "__main__":
    main()
