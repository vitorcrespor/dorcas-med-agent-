from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
import schema
from dotenv import load_dotenv
import os

load_dotenv()
LOG_PATH= os.getenv('LOG_PATH')
TOOL_LOG_PATH= os.getenv('TOOL_LOG_PATH')
SUMMARY_KEEP_MESSAGES = 6
SUMMARY_TRIGGER_MESSAGES = 10

def log(conversation_history: list[BaseMessage]= []):
    """   with open(TOOL_LOG_PATH, "w") as file:
        file.write("tool log\n")
        for message in tool_history:
            file.write(f"Tool: {message.name}\n")
            file.write(f"Tool call id: {message.tool_call_id}\n")
            file.write(f"Result: {message.content}\n\n")
        file.write("log end\n")"""
        
    with open(LOG_PATH, "w") as file:
        file.write("conversation log\n")
        for message in conversation_history:
            if isinstance(message, HumanMessage):
                file.write(f"USER: {message.content}\n")
            elif isinstance(message, AIMessage):
                file.write(f"DORCAS: {message.content}\n")
        file.write("log end\n")

def message_to_text(message) -> str:
    """Convert a LangChain message into compact text for summarization."""
    content= getattr(message, "content", "") or ""
    content= str(content).strip()
    
    if isinstance(message, HumanMessage): role= "USER"
    elif isinstance(message, AIMessage):
        role= "DORCAS"
        if getattr(message, "tool_calls", None):
            tool_names= [tc["name"] for tc in message.tool_calls]
            tool_queries = [tc.get("args", {}) for tc in message.tool_calls]
            content= f"Requested tool calls: {tool_names} | {tool_queries}"

    elif isinstance(message, ToolMessage):
        role= f"Tool({message.name})"
        content= content[:1500]
    else:
        role= type(message).__name__

    if not content: return ""
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
        return {"summary": old_summary}

    messages_to_summarize = messages[:-SUMMARY_KEEP_MESSAGES]
    transcript_parts = []

    for message in messages_to_summarize:
        text = message_to_text(message)
        if text:
            transcript_parts.append(text)
    transcript = "\n\n".join(transcript_parts)
    if not transcript.strip():
        return {"summary": old_summary}

    system_prompt = SystemMessage(content="""
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
                            """)

    human_prompt = HumanMessage(content=f"""Previous summary:{old_summary}
                            New conversation section to merge into the summary:
                            {transcript} 
                            Updated summary:
                            """)

    response= schema.lm.invoke([system_prompt,human_prompt])
    return {"summary": response.content.strip()}


def log_ingestion(conversation_history= []) -> list[BaseMessage]:
    with open(LOG_PATH,'r') as file:
        for line in file:
            if line.startswith("DORCAS"):
                conversation_history.append(AIMessage(content=line.removeprefix("DORCAS:").strip()))
            if line.startswith("USER"):
                conversation_history.append(AIMessage(content=line.removeprefix("DORCAS:").strip()))
    return conversation_history
