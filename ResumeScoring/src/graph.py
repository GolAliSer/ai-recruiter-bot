"""LangGraph: построение и компиляция графа обработки резюме"""

from langgraph.graph import StateGraph, END

from src.state import ResumeState
from src.nodes.parser import parser_node
from src.nodes.scorer import scorer_node
from src.nodes.role_advisor import suggest_role
from src.nodes.formatter import formatter_node


def build_graph():
    """Строит граф обработки резюме (4 узла: Parser → Scorer → Role Advisor → Formatter)"""
    workflow = StateGraph(ResumeState)

    workflow.add_node("parser", parser_node)
    workflow.add_node("scorer", scorer_node)
    workflow.add_node("role_advisor", suggest_role)
    workflow.add_node("formatter", formatter_node)

    workflow.set_entry_point("parser")
    workflow.add_edge("parser", "scorer")
    workflow.add_edge("scorer", "role_advisor")
    workflow.add_edge("role_advisor", "formatter")
    workflow.add_edge("formatter", END)

    return workflow.compile()


app_graph = build_graph()