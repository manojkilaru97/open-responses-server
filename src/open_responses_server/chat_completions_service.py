import json
from fastapi import Request
from fastapi.responses import StreamingResponse, Response, JSONResponse
from open_responses_server.common.llm_client import LLMClient
from open_responses_server.common.config import logger, OPENAI_BASE_URL_INTERNAL, MAX_TOOL_CALL_ITERATIONS, ENABLE_MCP_TOOLS
from open_responses_server.common.mcp_manager import mcp_manager, serialize_tool_result

async def _handle_non_streaming_request(client: LLMClient, request_data: dict):
    """Handles a non-streaming chat completions request - simple passthrough."""
    current_request_data = request_data.copy()
    
    # Remove reasoning parameter if it has null values
    if "reasoning" in current_request_data:
        reasoning = current_request_data["reasoning"]
        if isinstance(reasoning, dict) and all(v is None for v in reasoning.values()):
            current_request_data.pop("reasoning", None)
            logger.info("[CHAT-COMPLETIONS-NON-STREAM] Removed reasoning parameter with null values")
    
    # Make a single request to vLLM and return the result
    current_request_data.pop("stream", None)

    try:
        response = await client.post(
            "/v1/chat/completions",
            json=current_request_data,
            timeout=120.0
        )
        response.raise_for_status()
        response_data = response.json()
        
        choice = response_data.get("choices", [])[0]
        
        if choice.get("finish_reason") == "tool_calls":
            tool_calls = choice.get("message", {}).get("tool_calls", [])
            logger.info(f"[CHAT-COMPLETIONS-NON-STREAM] Returning {len(tool_calls)} tool calls to client for execution")
        
        # Return the response data directly - let client handle tool execution
        return response_data

    except Exception as e:
        logger.error(f"Error during non-streaming chat completion: {e}")
        return {"error": str(e)}


async def _handle_streaming_request(client: LLMClient, request_data: dict) -> StreamingResponse:
    """Handles a streaming chat completions request - simple passthrough."""
    stream_request_data = request_data.copy()
    stream_request_data["stream"] = True

    # Remove reasoning parameter if it has null values
    if "reasoning" in stream_request_data:
        reasoning = stream_request_data["reasoning"]
        if isinstance(reasoning, dict) and all(v is None for v in reasoning.values()):
            stream_request_data.pop("reasoning", None)
            logger.info("[CHAT-COMPLETIONS-STREAM] Removed reasoning parameter with null values")

    # Just proxy the stream directly - no tool call processing
    async def stream_proxy():
        try:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=stream_request_data,
                timeout=120.0
            ) as stream_response:
                async for chunk in stream_response.aiter_bytes():
                    yield chunk
        except Exception as e:
            logger.error(f"Error during chat completions stream proxy: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()

    return StreamingResponse(stream_proxy(), media_type="text/event-stream")


async def handle_chat_completions(request: Request):
    """
    Handles requests to the /v1/chat/completions endpoint.
    Injects MCP tools and proxies the request to the underlying LLM API.
    """
    client = await LLMClient.get_client()
    request_data = await request.json()

    logger.info("[CHAT-COMPLETIONS] Processing /v1/chat/completions request")
    
    # Inject MCP tools into the request
    mcp_tools = mcp_manager.get_mcp_tools()
    if mcp_tools and ENABLE_MCP_TOOLS:
        existing_tools = request_data.get("tools", [])
        existing_tool_names = {tool.get("function", {}).get("name") for tool in existing_tools}
        
        logger.info(f"[CHAT-COMPLETIONS] Found {len(mcp_tools)} MCP tools available")
        logger.info(f"[CHAT-COMPLETIONS] Request has {len(existing_tools)} existing tools: {list(existing_tool_names)}")
        
        added_tools = []
        for tool in mcp_tools:
            if tool.get("name") not in existing_tool_names:
                existing_tools.append({"type": "function", "function": tool})
                added_tools.append(tool.get("name"))
        
        request_data["tools"] = existing_tools
        
        logger.info(f"[CHAT-COMPLETIONS] Added {len(added_tools)} MCP tools: {added_tools}")
        logger.info(f"[CHAT-COMPLETIONS] Final tool count: {len(existing_tools)}")
    else:
        logger.info("[CHAT-COMPLETIONS] No MCP tools available to inject or MCP tools disabled.")
    
    logger.debug(f"[CHAT-COMPLETIONS] Final tools in request: {request_data.get('tools', [])}")

    # Determine if the request is streaming
    is_stream = request_data.get("stream", False)
    logger.info(f"[CHAT-COMPLETIONS] Request streaming mode: {is_stream}")

    if is_stream:
        return await _handle_streaming_request(client, request_data)
    else:
        return await _handle_non_streaming_request(client, request_data) 