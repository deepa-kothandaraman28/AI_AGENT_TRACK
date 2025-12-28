
import pandas as pd
from typing import TypedDict
from langgraph.graph import StateGraph
from transformers import pipeline
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

# ---------- Agent State ----------
class AgentState(TypedDict):
    event: str
    severity: int
    decision: str

llm_pipeline = pipeline("text-generation", model="sshleifer/tiny-gpt2")  

def local_llm_reasoning(prompt: str) -> str:
    output = llm_pipeline(prompt, max_length=20, do_sample=True)
    text = output[0]['generated_text'].strip()
    # Simple mapping to ESCALATE/ALERT/IGNORE based on keywords
    text = text.upper()
    if "ESCALATE" in text:
        return "ESCALATE"
    elif "ALERT" in text:
        return "ALERT"
    else:
        return "IGNORE"

# ---------- Reasoning Node ----------
def reasoning_agent(state: AgentState):
    prompt = f"Event: {state['event']}, Severity: {state['severity']}. Decide action: ESCALATE, ALERT, or IGNORE."
    decision = local_llm_reasoning(prompt)
    return {"decision": decision}

# ---------- Action Node ----------
def action_executor(state: AgentState):
    decision = state["decision"]

    if "ESCALATE" in decision:
        print("🚨 ESCALATING:", state["event"])
    elif "ALERT" in decision:
        print("⚠️ ALERT:", state["event"])
    else:
        print("✅ IGNORE:", state["event"])

    return state

# ---------- LangGraph ----------
graph = StateGraph(AgentState)
graph.add_node("reason", reasoning_agent)
graph.add_node("act", action_executor)
graph.set_entry_point("reason")
graph.add_edge("reason", "act")
graph.set_finish_point("act")
agent_app = graph.compile()
processed_events = set()

class CSVHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("live_events.csv"):
            data = pd.read_csv(event.src_path)
            for idx, row in data.iterrows():
                if idx not in processed_events:
                    state = {"event": row["event"], "severity": row["severity"]}
                    result = agent_app.invoke(state)
                    action_executor(result)
                    processed_events.add(idx)
observer = Observer()
observer.schedule(CSVHandler(), path="data", recursive=False)
observer.start()
print("🚀 Agent is watching live_events.csv for new events...")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()