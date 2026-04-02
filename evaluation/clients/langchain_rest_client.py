"""
Reference Client: LangChain REST API Integration
================================================

File:        langchain_rest_client.py
Purpose:     Demonstrates an external agent dynamically discovering tools
             from the AgentDisrupt-Bench Sandbox via HTTP, generating an OpenAPI-driven
             ReAct agent with full schema-aware tool binding.
"""

import httpx
import logging
from typing import Any, Callable, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger("agentdisruptbench.clients.langchain_rest")


def _resolve_ref(ref: str, spec: dict) -> dict:
    """Resolve a $ref pointer like '#/components/schemas/Foo' into the actual schema dict."""
    parts = ref.lstrip("#/").split("/")
    node = spec
    for part in parts:
        node = node[part]
    return node


def _openapi_type_to_python(prop: dict) -> type:
    """Map an OpenAPI type string to a Python type."""
    t = prop.get("type", "string")
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }
    return mapping.get(t, str)


def _build_pydantic_model(name: str, schema: dict, spec: dict) -> type[BaseModel]:
    """Build a Pydantic model from an OpenAPI request body schema."""
    # Resolve $ref if needed
    if "$ref" in schema:
        schema = _resolve_ref(schema["$ref"], spec)

    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    fields = {}
    for field_name, field_spec in properties.items():
        python_type = _openapi_type_to_python(field_spec)
        description = field_spec.get("title", field_name)
        default = field_spec.get("default")

        if field_name in required_fields:
            if default is not None:
                fields[field_name] = (python_type, Field(default=default, description=description))
            else:
                fields[field_name] = (python_type, Field(..., description=description))
        else:
            fields[field_name] = (Optional[python_type], Field(default=default, description=description))

    model = create_model(name, **fields)
    return model


def build_langchain_client(server_url: str) -> Callable[[str], str]:
    """Returns a callable agent function that takes (instruction, server_url)."""

    # 1. Fetch the OpenAPI Spec
    try:
        resp = httpx.get(f"{server_url}/openapi.json", timeout=10.0)
        if resp.status_code != 200:
            raise ConnectionError(f"Could not load OpenAPI from {server_url}")
    except httpx.RequestError as e:
        raise ConnectionError(f"Could not connect to {server_url}: {e}") from e
    openapi_spec = resp.json()

    # 2. Dynamically build LangChain tools from the OpenAPI spec with full schemas
    tools = []

    for path, path_item in openapi_spec.get('paths', {}).items():
        # Only expose tool endpoints, skip admin routes
        if not path.startswith("/api/tools/"):
            continue
        if 'post' not in path_item:
            continue

        op = path_item['post']
        tool_name = path.split('/')[-1]
        desc = op.get('description', op.get('summary', tool_name))

        # Build arg schema from the request body
        args_schema = None
        req_body = op.get('requestBody', {})
        content = req_body.get('content', {}).get('application/json', {})
        body_schema = content.get('schema', {})
        if body_schema:
            try:
                model_name = f"{tool_name}_args"
                args_schema = _build_pydantic_model(model_name, body_schema, openapi_spec)
            except Exception as e:
                logger.warning(f"Could not build schema for {tool_name}: {e}")

        def make_caller(endpoint: str):
            def call_api(**kwargs):
                r = httpx.post(f"{server_url}{endpoint}", json=kwargs, timeout=30.0)
                if r.status_code == 200:
                    return r.json()
                return {"error": r.text, "status_code": r.status_code}
            return call_api

        tool_kwargs = dict(
            func=make_caller(path),
            name=tool_name,
            description=desc,
        )
        if args_schema:
            tool_kwargs["args_schema"] = args_schema

        tool = StructuredTool.from_function(**tool_kwargs)
        tools.append(tool)

    logger.info(f"Dynamically mapped {len(tools)} tools from Sandbox Server")

    if not tools:
        raise ConnectionError(f"No tools found in the OpenAPI spec at {server_url}")

    # 3. Setup the ReAct Agent using Gemini and LangGraph
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    agent = create_react_agent(llm, tools)

    # 4. Define the execution function
    def run_agent(instruction: str) -> str:
        try:
            state = {"messages": [("user", instruction)]}
            result_state = agent.invoke(state)
            if result_state and "messages" in result_state:
                final = result_state["messages"][-1].content
                # Normalize Gemini list-of-parts format
                if isinstance(final, list):
                    parts = []
                    for item in final:
                        if isinstance(item, dict):
                            parts.append(item.get("text", str(item)))
                        else:
                            parts.append(str(item))
                    return "\n".join(parts)
                return str(final) if final else ""
        except Exception as e:
            logger.error(f"Agent Execution Failed: {e}")
        return ""

    return run_agent
