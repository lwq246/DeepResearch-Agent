from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from .nodes import generate
from .nodes import react_plan
from .nodes import retrieve
from .nodes import route_after_validation
from .nodes import route_react_action
from .nodes import validate_evidence
from .nodes import web_search
from .state import GraphState


load_dotenv(Path(__file__).resolve().parent / ".env")


workflow = StateGraph(GraphState)
workflow.add_node("react_plan", react_plan)
workflow.add_node("retrieve", retrieve)
workflow.add_node("web_search", web_search)
workflow.add_node("validate_evidence", validate_evidence)
workflow.add_node("generate", generate)
workflow.set_entry_point("react_plan")
workflow.add_conditional_edges("react_plan", route_react_action)
workflow.add_edge("retrieve", "validate_evidence")
workflow.add_edge("web_search", "validate_evidence")
workflow.add_conditional_edges("validate_evidence", route_after_validation)
workflow.add_edge("generate", END)

app = workflow.compile()
