import logging
import os
import sys
import time
import uuid
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

# Ensure local 'evaluation' python directory is importable
local_eval_path = os.path.abspath(os.path.join(os.getcwd()))
if local_eval_path not in sys.path:
    sys.path.insert(0, local_eval_path)

from evaluation.clients.langchain_rest_client import build_langchain_client  # noqa: E402

logger = logging.getLogger("agentdisruptbench.proxy")

app = FastAPI(
    title="AgentDisruptBench OpenAI Proxy",
    description="A proxy that turns standard LLM chat completion requests into full Sandbox Agent evaluations.",
    version="1.0",
)

# ─── OpenAI Compatible Schemas ───


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False


class ChoiceMessage(BaseModel):
    role: str
    content: str


class Choice(BaseModel):
    index: int
    message: ChoiceMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "AgentDisruptBench Proxy Server is running! Point your OpenAI clients to this host with base_url='/v1'.",
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    """
    Intercept an OpenAI compatible chat completion.
    The last message content is treated as the Benchmark Task Instruction.
    We deploy the LangChain Sandbox reference agent to solve it, and return the final string!
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty.")

    # Extract the user instruction from the latest message
    last_message = req.messages[-1].content

    logger.info(f"Received proxy execution request for model {req.model}. Launching Sandbox Agent.")

    # Read sandbox URL from env var (set by CLI) or fall back to default
    sandbox_url = os.environ.get("ADB_SANDBOX_URL", "http://localhost:8080")

    try:
        agent_runner = build_langchain_client(sandbox_url)
    except Exception as e:
        logger.error(f"Failed to initialize target agent mapping: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize Sandbox Agent client.")

    try:
        agent_output = agent_runner(last_message, sandbox_url)
    except Exception as e:
        logger.error(f"Agent Loop crashed during proxy execution: {e}")
        agent_output = f"Execution failed locally: {str(e)}"

    # Normalize output — Gemini can return list-of-parts instead of plain string
    if isinstance(agent_output, list):
        parts = []
        for item in agent_output:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        agent_output = "\n".join(parts)
    if not isinstance(agent_output, str):
        agent_output = str(agent_output)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    response = ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=req.model,
        choices=[Choice(index=0, message=ChoiceMessage(role="assistant", content=agent_output), finish_reason="stop")],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )

    return response


if __name__ == "__main__":
    import uvicorn

    # Typically run via `adb proxy --port 8082`
    uvicorn.run("proxy:app", host="0.0.0.0", port=8082, reload=True)
