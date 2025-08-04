import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from open_responses_server.common.config import logger
from open_responses_server.common.config import ENABLE_MCP_TOOLS
from open_responses_server.common.llm_client import startup_llm_client, shutdown_llm_client, LLMClient
from open_responses_server.common.mcp_manager import mcp_manager
from open_responses_server.responses_service import convert_responses_to_chat_completions, process_chat_completions_stream, convert_chat_completions_to_responses
from open_responses_server.chat_completions_service import handle_chat_completions

app = FastAPI(
    title="Open Responses Server",
    description="A proxy server that converts between different OpenAI-compatible API formats.",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Application startup event handler."""
    await startup_llm_client()
    await mcp_manager.startup_mcp_servers()
    logger.info("API Controller startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event handler."""
    await shutdown_llm_client()
    await mcp_manager.shutdown_mcp_servers()
    logger.info("API Controller shutdown complete.")


# API endpoints
@app.post("/responses")
async def create_response(request: Request):
    """
    Create a response in Responses API format, translating to/from chat.completions API.
    """
    try:
        logger.info("Received request to /responses")
        request_data = await request.json()
        
        # Log basic request information
        logger.info(f"Received request: model={request_data.get('model')}, stream={request_data.get('stream')}")
        
        # Log basic input summary
        if "input" in request_data and request_data["input"]:
            input_count = len(request_data["input"])
            logger.info(f"Processing {input_count} input item(s)")
        
        # Inject cached MCP tools into request_data before conversion so conversion sees them
        if mcp_manager.mcp_functions_cache and ENABLE_MCP_TOOLS:
            # Get existing tools from request_data or initialize empty list
            existing_tools = request_data.get("tools", [])
            
            # Create tools format for MCP functions
            mcp_tools = [
                {"type": "function", "name": f["name"], "description": f.get("description"), "parameters": f.get("parameters", {})}
                for f in mcp_manager.mcp_functions_cache
            ]
            
            # Get the names of existing tools to avoid duplicates
            existing_tool_names = set(tool["name"] for tool in existing_tools if "name" in tool)
            
            # Only add MCP tools that don't conflict with existing tools
            filtered_mcp_tools = [
                tool for tool in mcp_tools 
                if tool["name"] not in existing_tool_names
            ]
            
            # Append filtered MCP tools to existing tools, keeping existing tools first (priority)
            request_data["tools"] = existing_tools + filtered_mcp_tools
            
            logger.info(f"Injected {len(filtered_mcp_tools)} MCP tools into request (total: {len(request_data['tools'])} tools)")
        else:
            logger.info("No MCP tools available or disabled")
        
        # Convert request to chat.completions format
        chat_request = convert_responses_to_chat_completions(request_data)
        
        # Inject cached MCP tool definitions
        if mcp_manager.mcp_functions_cache and ENABLE_MCP_TOOLS:
            # Keep any existing functions and merge with MCP functions
            existing_functions = chat_request.get("functions", [])
            
            # Convert to the "tools" format which is more broadly supported
            if "tools" not in chat_request:
                chat_request["tools"] = []
                
            # Get existing tool names to avoid duplicates and ensure priority
            existing_tool_names = set()
            for tool in chat_request["tools"]:
                if isinstance(tool, dict) and "function" in tool and "name" in tool["function"]:
                    existing_tool_names.add(tool["function"]["name"])
                elif isinstance(tool, dict) and "name" in tool:
                    existing_tool_names.add(tool["name"])
            
            # First convert existing functions to tools format
            for func in existing_functions:
                if func.get("name") not in existing_tool_names:
                    chat_request["tools"].append({
                        "type": "function",
                        "function": func
                    })
                    existing_tool_names.add(func.get("name", ""))
            
            # Then add MCP functions that don't conflict with existing tools
            mcp_tools_added = []
            for func in mcp_manager.mcp_functions_cache:
                if func.get("name") not in existing_tool_names:
                    chat_request["tools"].append({
                        "type": "function",
                        "function": func
                    })
                    mcp_tools_added.append(func.get("name"))
            
            # Remove the functions key as we've converted to tools format
            chat_request.pop("functions", None)
            
            logger.info(f"Converted {len(existing_functions)} functions, added {len(mcp_tools_added)} MCP tools to chat request")
        else:
            logger.info("No MCP functions cached or MCP tools disabled")
        
        # Remove tool_choice when no functions/tools are provided
        if not chat_request.get("functions") and not chat_request.get("tools"):
            chat_request.pop("tool_choice", None)
        # End MCP injection
        # Remove unsupported tool_choice parameter before sending
        chat_request.pop("tool_choice", None)

        # Check for streaming mode
        stream = request_data.get("stream", False)
        
        if stream:
            logger.info("Handling streaming response")
            # Handle streaming response
            async def stream_response():
                try:
                    logger.info(f"Sending chat completions request with {len(chat_request.get('messages', []))} messages")
                    client = await LLMClient.get_client()
                    async with client.stream(
                        "POST",
                        "/v1/chat/completions",
                        json=chat_request,
                        timeout=120.0
                    ) as response:
                        logger.info(f"Stream request status: {response.status_code}")
                        
                        if response.status_code != 200:
                            error_content = await response.aread()
                            logger.error(f"Error from LLM API: {error_content}")
                            yield f"data: {json.dumps({'type': 'error', 'error': {'message': f'Error from LLM API: {response.status_code}'}})}\n\n"
                            return
                        
                        async for event in process_chat_completions_stream(response, chat_request):
                            yield event
                except Exception as e:
                    logger.error(f"Error in stream_response: {str(e)}")
                    yield f"data: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"
            
            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream"
            )
        
        else:
            logger.info("Handling non-streaming response")
            # Handle non-streaming response
            try:
                client = await LLMClient.get_client()
                response = await client.post(
                    "/v1/chat/completions",
                    json=chat_request,
                    timeout=120.0
                )
                response.raise_for_status()
                response_data = response.json()
                
                # Convert chat completions response to responses API format
                responses_response = convert_chat_completions_to_responses(response_data, chat_request)
                return JSONResponse(content=responses_response)
                
            except Exception as e:
                logger.error(f"Error in non-streaming response: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing non-streaming request: {str(e)}"
                )
            
    except Exception as e:
        logger.error(f"Error in create_response: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Endpoint for /v1/chat/completions, delegating to the service.
    """
    logger.info("Handling chat completions")
    response = await handle_chat_completions(request)
    logger.info("Chat completions handled")
    if isinstance(response, StreamingResponse):
        return response
    elif isinstance(response, Response):
        return response
    return response


@app.get("/health")
async def health_check():
    """Return health of adapter and underlying LLM backend."""
    from open_responses_server.common.llm_client import LLMClient
    llm_up = False
    try:
        client = await LLMClient.get_client()
        resp = await client.get("/v1/models", timeout=5.0)
        llm_up = resp.status_code == 200
    except Exception:
        llm_up = False
    overall = "ok" if llm_up else "degraded"
    return {"status": overall, "adapter": "running", "llm_backend": llm_up}

@app.get("/")
async def root():
    return {"message": "Open Responses Server is running."}

@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH", "TRACE"])
async def proxy_endpoint(request: Request, path_name: str):
    """
    A generic proxy for any other endpoints, forwarding them to the LLM backend.
    """
    client = await LLMClient.get_client()
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() != 'host'}

    try:
        url = f"{client.base_url}/v1/{path_name}"
        
        # Handle streaming for the proxy
        is_stream = False
        if body:
            try:
                is_stream = json.loads(body).get("stream", False)
            except json.JSONDecodeError:
                pass

        if is_stream:
            async def stream_proxy():
                async with client.stream(request.method, url, headers=headers, content=body, timeout=120.0) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
            return StreamingResponse(stream_proxy(), media_type=request.headers.get('accept', 'application/json'))
        else:
            response = await client.request(request.method, url, headers=headers, content=body, timeout=120.0)
            return Response(content=response.content, status_code=response.status_code, headers=response.headers)
            
    except Exception as e:
        logger.error(f"Error in proxy endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error proxying request: {str(e)}")