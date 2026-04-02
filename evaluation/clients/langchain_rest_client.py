"""
Reference Client: LangChain REST API Integration
================================================

File:        langchain_rest_client.py
Purpose:     Demonstrates an external agent dynamically discovering tools
             from the AgentDisrupt-Bench Sandbox via HTTP, generating an OpenAPI-driven
             ReAct agent.
"""

import httpx
import logging
from typing import Callable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_core.tools import StructuredTool
from pydantic import create_model

logger = logging.getLogger("agentdisruptbench.clients.langchain_rest")

def build_langchain_client(server_url: str) -> Callable[[str, str], bool]:
    """Returns a callable agent function that takes (instruction, server_url)."""
    
    # 1. Fetch the OpenAPI Spec
    resp = httpx.get(f"{server_url}/openapi.json")
    if resp.status_code != 200:
        raise ConnectionError(f"Could not load OpenAPI from {server_url}")
    openapi_spec = resp.json()

    # 2. Dynamically build LangChain tools from the OpenAPI spec
    tools = []
    
    for path, path_item in openapi_spec.get('paths', {}).items():
        if 'post' not in path_item:
            continue
            
        op = path_item['post']
        tool_name = path.split('/')[-1]
        desc = op.get('summary', tool_name)
        
        # We define a generic caller that shoots POST to the path
        def make_caller(endpoint: str):
            def call_api(**kwargs):
                r = httpx.post(f"{server_url}{endpoint}", json=kwargs)
                if r.status_code == 200:
                    return r.json()
                return {"error": r.text, "status_code": r.status_code}
            return call_api
            
        # Create a dynamic pydantic schema based on OpenAPI components
        # (For absolute robustness in this script we use a generic dict or try to map)
        # Langchain allows typing to just **kwargs for flexible tools if needed.
        
        tool = StructuredTool.from_function(
            func=make_caller(path),
            name=tool_name,
            description=desc,
        )
        tools.append(tool)

    logger.info(f"Dynamically mapped {len(tools)} tools from Sandbox Server")

    # 3. Setup the ReAct Agent using Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # Using a standard ReAct prompt from Langchain Hub or simple system message
    from langchain import hub
    try:
        prompt = hub.pull("hwchase17/react-chat-json")
    except Exception:
        # Fallback if LangSmith isn't configured
        from langchain.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant with access to the following tools: {tools}\n{tool_names}\nUse them to fulfill the user request."),
            ("human", "{input}\n{agent_scratchpad}")
        ])

    agent = create_json_chat_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # 4. Define the execution function
    def run_agent(instruction: str, url: str) -> bool:
        try:
            result = executor.invoke({"input": instruction})
            # Crude success criteria: it didn't throw an exception and generated output
            if result and "output" in result:
                return True
        except Exception as e:
            logger.error(f"Agent Execution Failed: {e}")
        return False

    return run_agent
