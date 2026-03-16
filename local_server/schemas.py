# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Any, Dict, Union


# ----------------------------
# /generate
# ----------------------------
class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, examples=["Explain quantum physics in 5-year-old terms."])
    max_tokens: int = Field(256, ge=10, le=2048, examples=[128])
    temperature: float = Field(0.7, ge=0.0, le=2.0, examples=[0.7])


class GenerationResponse(BaseModel):
    result: str
    token_usage: int


# ----------------------------
# /v1/chat/completions
# ----------------------------
class ChatMessage(BaseModel):
    # "tool" role is required for Agents SDK tool-result messages
    role: Literal["system", "user", "assistant", "developer", "tool"]
    content: Optional[str] = None
    tool_call_id: Optional[str] = None   # present on role="tool" messages
    name: Optional[str] = None           # optional tool name hint


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = "text"


class ChatCompletionRequest(BaseModel):
    model: str = "local-model"
    messages: List[ChatMessage]
    max_tokens: int = 512          # bumped from 256 — agents need more room
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False
    response_format: Optional[ResponseFormat] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = "auto"


class ChatCompletionChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionChoiceMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, Any] = {}


# ----------------------------
# Streaming chunk schemas
# (used when stream=True)
# ----------------------------
class ChunkDelta(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


# ----------------------------
# /v1/embeddings (OpenAI-compatible)
# ----------------------------
class EmbeddingsRequest(BaseModel):
    model: str = "local-embed"
    input: Union[str, List[str]]


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: List[float]
    index: int


class EmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, Any] = {}
