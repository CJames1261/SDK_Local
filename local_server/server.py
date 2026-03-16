# server.py  (local_server_v2 — OpenAI Agents SDK compatible)
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import time
import uuid
import json
from typing import List, Generator

from ml_engine import llm_engine
from schemas import (
    GenerationRequest,
    GenerationResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingData,
)

# Keywords that identify mem0 graph store calls which expect JSON but don't
# set response_format=json_object
_GRAPH_JSON_KEYWORDS = [
    "entities and their types",
    "knowledge graph",
    "graph memory",
    "extract structured information from text",
]


def _needs_json_enforcement(messages: list) -> bool:
    """Return True if this is a graph-store call that expects JSON output."""
    for m in messages:
        if m.get("role") == "system":
            content = (m.get("content") or "").lower()
            if any(kw in content for kw in _GRAPH_JSON_KEYWORDS):
                return True
    return False


# ----------------------------
# SSE helpers
# ----------------------------
def _sse(chunk_id: str, model_name: str, delta: dict, finish_reason=None) -> str:
    """Serialise a single SSE data line for a chat completion chunk."""
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _stream_text(chunk_id: str, model_name: str, text: str) -> Generator[str, None, None]:
    """Yield SSE events for a plain-text response."""
    yield _sse(chunk_id, model_name, {"role": "assistant", "content": ""})
    yield _sse(chunk_id, model_name, {"content": text})
    yield _sse(chunk_id, model_name, {}, finish_reason="stop")
    yield "data: [DONE]\n\n"


def _stream_tool_calls(
    chunk_id: str,
    model_name: str,
    tool_calls_payload: list,
    finish_reason: str,
) -> Generator[str, None, None]:
    """Yield SSE events for one or more parallel tool calls.

    For each tool call we emit two chunks:
      1. Header chunk — id, type, function name, empty arguments string.
         The very first header also carries role="assistant".
      2. Arguments chunk — full arguments string delta.
    Then a final finish chunk.
    """
    for i, tc in enumerate(tool_calls_payload):
        # Header chunk — only the first carries role="assistant"
        delta: dict = {
            "tool_calls": [{
                "index": i,
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["function"]["name"], "arguments": ""},
            }]
        }
        if i == 0:
            delta["role"] = "assistant"
            delta["content"] = None
        yield _sse(chunk_id, model_name, delta)

        # Arguments chunk
        yield _sse(chunk_id, model_name, {
            "tool_calls": [{
                "index": i,
                "function": {"arguments": tc["function"]["arguments"]},
            }],
        })

    # Final chunk — signals end of the turn
    yield _sse(chunk_id, model_name, {}, finish_reason=finish_reason)
    yield "data: [DONE]\n\n"


# ----------------------------
# App
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    llm_engine.load_model()
    llm_engine.load_embedder()
    app.title = f"Local Mistral API v2 (SDK) — {llm_engine.model_name}"
    yield
    print("🛑 Shutting down...")


app = FastAPI(lifespan=lifespan)


# ----------------------------
# Root / health
# ----------------------------
@app.get("/")
def read_root():
    return {
        "status": "online",
        "model": llm_engine.model_name,
        "embedder": getattr(llm_engine, "embed_model_name", None),
        "version": "v2 — OpenAI Agents SDK compatible",
        "streaming": "supported",
    }


# ----------------------------
# /generate  (unchanged from v1)
# ----------------------------
@app.post("/generate", response_model=GenerationResponse)
def generate_text(request: GenerationRequest):
    try:
        result = llm_engine.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return GenerationResponse(result=result, token_usage=len(result.split()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# OpenAI-like /v1/models
# ----------------------------
@app.get("/v1/models")
def list_models():
    model_name = llm_engine.model_name or "local-model"
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


# ----------------------------
# OpenAI-like /v1/chat/completions
# Supports stream=True via SSE
# ----------------------------
@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    try:
        model_name = llm_engine.model_name or (req.model or "local-model")
        messages = [m.model_dump() for m in req.messages]
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"

        # ── TOOL CALLING PATH ─────────────────────────────────────────────────
        # The Agents SDK sends tools=[...] when it needs function/tool calling.
        # We instruct the LLM to output the matching JSON schema, then wrap the
        # result in the tool_calls wire format the SDK expects.
        #
        # IMPORTANT: if the conversation already contains a role="tool" message,
        # the SDK is asking the orchestrator to synthesize a final answer from
        # the tool results — NOT to call another tool.  Force TEXT MODE so the
        # LLM produces a plain-text response instead of more JSON.
        has_tool_results = any(m.get("role") == "tool" for m in messages)

        if req.tools and not has_tool_results:
            # Build a summary of every available tool so the LLM can choose
            tools_summary = "\n".join(
                f'- {t["function"]["name"]}: {t["function"].get("description", "")}'
                for t in req.tools
                if "function" in t
            )

            # Ask the LLM to output a JSON ARRAY so it can call multiple tools
            # in a single turn (e.g. "translate in French AND Spanish").
            selection_instruction = (
                "You MUST respond with ONLY a valid JSON array and nothing else. "
                "No markdown, no code fences, no explanation.\n\n"
                "Available tools:\n"
                f"{tools_summary}\n\n"
                "For EACH action the user requests, include one entry. "
                "If the user asks for multiple languages, include multiple entries.\n"
                "Use exactly this structure:\n"
                '[{"tool_name": "<name of tool>", "arguments": {"input": "<text to process>"}}, ...]'
            )

            if messages and messages[0]["role"] == "system":
                messages[0] = {
                    "role": "system",
                    "content": selection_instruction + "\n\n" + (messages[0].get("content") or ""),
                }
            else:
                messages = [{"role": "system", "content": selection_instruction}] + messages

            print(f"\n[TOOL SELECTION] messages sent to LLM:")
            for m in messages:
                print(f"  [{m['role']}]: {str(m.get('content', ''))[:400]}")

            text = llm_engine.generate_chat(
                messages=messages,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )
            print(f"[TOOL SELECTION] LLM raw response: {text[:600]}")

            # Parse JSON array; retry at temperature=0 if malformed
            def _parse_tool_list(raw: str):
                parsed = json.loads(raw)
                # Normalise: a single dict is also accepted
                return parsed if isinstance(parsed, list) else [parsed]

            try:
                tool_list = _parse_tool_list(text)
            except Exception:
                retry_messages = list(messages)
                retry_messages[0] = {
                    "role": "system",
                    "content": "RETURN ONLY A VALID JSON ARRAY. No prose. No markdown.\n\n"
                               + (messages[0].get("content") or ""),
                }
                text = llm_engine.generate_chat(
                    messages=retry_messages,
                    max_new_tokens=req.max_tokens,
                    temperature=0.0,
                    top_p=req.top_p,
                )
                try:
                    tool_list = _parse_tool_list(text)
                except Exception:
                    tool_list = []

            # Build tool_calls_payload for every valid entry in the array
            tool_calls_payload = []
            for entry in tool_list:
                chosen_name = entry.get("tool_name", "")
                matched_tool = next(
                    (t for t in req.tools if t["function"]["name"] == chosen_name),
                    None,
                )
                if matched_tool is None:
                    continue
                arguments_obj = entry.get("arguments", {})
                required_fields = matched_tool["function"].get("parameters", {}).get("required", [])
                if required_fields and all(f in arguments_obj for f in required_fields):
                    tool_calls_payload.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": chosen_name,
                            "arguments": json.dumps(arguments_obj),
                        },
                    })

            finish_reason = "tool_calls" if tool_calls_payload else "stop"
            print(f"[TOOL SELECTED]: {[tc['function']['name'] for tc in tool_calls_payload]}")

            if req.stream:
                return StreamingResponse(
                    _stream_tool_calls(chunk_id, model_name, tool_calls_payload, finish_reason),
                    media_type="text/event-stream",
                )

            return ChatCompletionResponse(
                id=chunk_id,
                created=int(time.time()),
                model=model_name,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionChoiceMessage(
                            content=None,
                            tool_calls=tool_calls_payload or None,
                        ),
                        finish_reason=finish_reason,
                    )
                ],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        # ── JSON MODE / TEXT MODE PATH ────────────────────────────────────────
        # When the conversation contains tool results, the LLM's job is to
        # present those results to the user — not repeat the original question.
        # Append a synthesis instruction to the system prompt so the model
        # knows what to do with the [Tool result] messages it receives.
        if has_tool_results:
            synthesis_note = (
                "Tool results are now available in the conversation. "
                "Present the translated text(s) from the tool results to the user "
                "with a brief explanation labelling each language."
            )
            if messages and messages[0]["role"] == "system":
                messages[0] = {
                    "role": "system",
                    "content": (messages[0].get("content") or "") + "\n\n" + synthesis_note,
                }
            else:
                messages = [{"role": "system", "content": synthesis_note}] + messages

        rf_type = (req.response_format.type if req.response_format else "text")
        force_json = rf_type == "json_object" or _needs_json_enforcement(messages)

        if force_json:
            json_instruction = (
                "You MUST respond with a single valid JSON object and nothing else. "
                "Do not include markdown, code fences, or extra text. "
                "All strings must use double quotes. "
                "Do not use trailing commas. "
                "Return {} if you are unsure."
            )
            if messages and messages[0]["role"] == "system":
                messages[0] = {
                    "role": "system",
                    "content": json_instruction + "\n\n" + (messages[0].get("content") or ""),
                }
            else:
                messages = [{"role": "system", "content": json_instruction}] + messages

        label = "[JSON MODE]" if force_json else "[TEXT MODE]"
        print(f"\n{label} messages sent to LLM:")
        for m in messages:
            print(f"  [{m['role']}]: {str(m.get('content', ''))[:400]}")

        text = llm_engine.generate_chat(
            messages=messages,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        print(f"{label} LLM raw response: {text[:600]}")

        # Validate JSON and retry once if malformed
        if force_json:
            try:
                json.loads(text)
            except Exception:
                retry_instruction = (
                    "RETURN ONLY VALID JSON. No prose. No markdown. "
                    "If you output anything other than JSON, it is wrong."
                )
                if messages and messages[0]["role"] == "system":
                    retry_messages = list(messages)
                    retry_messages[0] = {
                        "role": "system",
                        "content": retry_instruction + "\n\n" + (messages[0].get("content") or ""),
                    }
                else:
                    retry_messages = [{"role": "system", "content": retry_instruction}] + messages
                text = llm_engine.generate_chat(
                    messages=retry_messages,
                    max_new_tokens=req.max_tokens,
                    temperature=0.0,
                    top_p=req.top_p,
                )
                json.loads(text)  # final check — propagates as 500 if still broken

        # ── Streaming text response ───────────────────────────────────────────
        if req.stream:
            return StreamingResponse(
                _stream_text(chunk_id, model_name, text),
                media_type="text/event-stream",
            )

        # ── Non-streaming text response ───────────────────────────────────────
        return ChatCompletionResponse(
            id=chunk_id,
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionChoiceMessage(content=text),
                    finish_reason="stop",
                )
            ],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# OpenAI-like /v1/embeddings
# ----------------------------
@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
def embeddings(req: EmbeddingsRequest):
    try:
        texts: List[str] = [req.input] if isinstance(req.input, str) else req.input
        vectors = llm_engine.embed_texts(texts)
        data = [EmbeddingData(embedding=vectors[i], index=i) for i in range(len(vectors))]
        return EmbeddingsResponse(
            data=data,
            model=req.model or llm_engine.embed_model_name,
            usage={},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
