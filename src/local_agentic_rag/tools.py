import json
from pydantic import BaseModel

class ToolCall(BaseModel):
    name: str
    parameters: dict

def parse_llm_action(text: str) -> ToolCall | None:
    try:
        clean = text.strip().removeprefix("```json").removesuffix("```").strip()
        return ToolCall(**json.loads(clean))
    except Exception:
        return None

TOOLS_SCHEMA = """
[
  {"name": "query_rag", "description": "Search knowledge base", "parameters": {"query": "string"}},
  {"name": "read_file", "description": "Read exact file", "parameters": {"filepath": "string"}}
]
"""