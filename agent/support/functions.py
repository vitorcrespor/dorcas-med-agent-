from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import schema, state

SUMMARY_KEEP_MESSAGES = 6
SUMMARY_TRIGGER_MESSAGES = 10


def message_to_text(message) -> str:
    """Convert a LangChain message into compact text for summarization."""
    content = getattr(message, "content", "") or ""
    content = str(content).strip()
    
    if isinstance(message, HumanMessage): role = "USER"
    elif isinstance(message, AIMessage):
        role = "DORCAS"
        if getattr(message, "tool_calls", None):
            tool_names = [tc["name"] for tc in message.tool_calls]
            content = f"Requested tool calls: {tool_names}"

    elif isinstance(message, ToolMessage):
        role = f"Tool({message.name})"
        # Avoid dumping huge RAG chunks into the summarizer
        content = content[:1500]
    else:
        role = type(message).__name__

    if not content:
        return ""
    return f"{role}: {content}"


def update_summary(state: schema.AgentState) -> schema.AgentState:
    """
    Summarize older conversation history into state['summary'].

    This does not generate the final user-facing answer.
    It only updates memory/context.
    """

    messages = list(state["messages"])
    old_summary = state.get("summary", "")

    if len(messages) <= SUMMARY_TRIGGER_MESSAGES:
        return {
            "summary": old_summary
        }

    messages_to_summarize = messages[:-SUMMARY_KEEP_MESSAGES]

    transcript_parts = []

    for message in messages_to_summarize:
        text = message_to_text(message)

        if text:
            transcript_parts.append(text)

    transcript = "\n\n".join(transcript_parts)

    if not transcript.strip():
        return {
            "summary": old_summary
        }

    system_prompt = SystemMessage(
        content="""
You summarize conversation history for an agent.

Create a compact but useful memory summary.

Preserve:
- user goals and preferences
- project architecture decisions
- unresolved bugs
- important code structure
- relevant retrieved document facts
- tool results if they matter

Do not include irrelevant greetings.
Do not invent details.
Return only the updated summary.
"""
    )

    human_prompt = HumanMessage(
        content=f"""
Previous summary:
{old_summary}

New conversation section to merge into the summary:
{transcript}

Updated summary:
"""
    )

    # Use the normal LLM, NOT the tool-bound rag_agent
    response = schema.lm.invoke([
        system_prompt,
        human_prompt,
    ])

    return {
        "summary": response.content.strip()
    }