from typing_extensions import TypedDict
from typing import Annotated
from langchain_groq import ChatGroq
from langgraph.graph import START,END,StateGraph,add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

class State(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

model = ChatGroq(model="llama-3.1-8b-instant")

def make_default_graph():
    graph_workflow = StateGraph(State)

    def call_model(state:State):
        return {"messages":[model.invoke(state["messages"])]}
    
    graph_workflow.add_node("agent",call_model)

    graph_workflow.add_edge(START,"agent")
    graph_workflow.add_edge("agent",END)

    agent = graph_workflow.compile()
    return agent

def make_toolcalling_graph():
    graph = StateGraph(State)

    arxiv_api = ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=500)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_api)

    wiki_api = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
    wiki = WikipediaQueryRun(api_wrapper=wiki_api)

    tavily = TavilySearchResults()

    #combine all these tools in the list
    tools = [arxiv, wiki, tavily]

    
    llm_with_tools = model.bind_tools(tools)

    def tool_calling_llm(state:State):
        return {"messages":[llm_with_tools.invoke(state["messages"])]}
    
    #add nodes to the graph
    graph.add_node("tool_calling_llm",tool_calling_llm)
    graph.add_node("tools",ToolNode(tools))

    #create edges for these nodes
    graph.add_edge(START,"tool_calling_llm")
    graph.add_conditional_edges("tool_calling_llm",tools_condition)
    graph.add_edge("tools","tool_calling_llm") #this edge makes normal graph into ReAct Graph

    #compile the graph
    graph_builder = graph.compile()
    return graph_builder

agent = make_toolcalling_graph()

