# ABOUTME: Single-file Temporal worker implementing a durable agentic loop with Google Gemini.
# Contains the workflow, activities, tool definitions, and worker setup all in one file.

import asyncio
import inspect
import json
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RawValue
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.envconfig import ClientConfig
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    # Import pydantic internals early to avoid sandbox warnings
    import pydantic_core  # noqa: F401
    import annotated_types  # noqa: F401

    import httpx
    from pydantic import BaseModel, Field
    from google import genai
    from google.genai import types


# =============================================================================
# System Instructions (Agent Personality)
# =============================================================================

SYSTEM_INSTRUCTIONS = """
You are a helpful agent that can use tools to help the user.
You will be given an input from the user and a list of tools to use.
You may or may not need to use the tools to satisfy the user ask.
If no tools are needed, respond in haikus.
"""


# =============================================================================
# Tool Definitions
# =============================================================================

# --- Weather Tool ---

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


class GetWeatherAlertsRequest(BaseModel):
    """Request model for getting weather alerts."""

    state: str = Field(description="Two-letter US state code (e.g. CA, NY)")


async def get_weather_alerts(request: GetWeatherAlertsRequest) -> str:
    """Get weather alerts for a US state.

    Args:
        request: The request object containing:
            - state: Two-letter US state code (e.g. CA, NY)
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    url = f"{NWS_API_BASE}/alerts/active/area/{request.state}"

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=5.0)
        response.raise_for_status()
        return json.dumps(response.json())


# --- Location Tools ---


class GetLocationRequest(BaseModel):
    """Request model for getting location info from an IP address."""

    ipaddress: str = Field(description="An IP address")


async def get_ip_address() -> str:
    """Get the public IP address of the current machine."""
    async with httpx.AsyncClient() as client:
        response = await client.get("https://icanhazip.com")
        response.raise_for_status()
        return response.text.strip()


async def get_location_info(request: GetLocationRequest) -> str:
    """Get the location information for an IP address including city, state, and country.

    Args:
        request: The request object containing:
            - ipaddress: An IP address to look up
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://ip-api.com/json/{request.ipaddress}")
        response.raise_for_status()
        result = response.json()
        return f"{result['city']}, {result['regionName']}, {result['country']}"


# =============================================================================
# Tool Registry
# =============================================================================

ToolHandler = Callable[..., Awaitable[Any]]


def get_handler(tool_name: str) -> ToolHandler:
    """Get the handler function for a given tool name."""
    if tool_name == "get_location_info":
        return get_location_info
    if tool_name == "get_ip_address":
        return get_ip_address
    if tool_name == "get_weather_alerts":
        return get_weather_alerts
    raise ValueError(f"Unknown tool name: {tool_name}")


def get_tools() -> types.Tool:
    """Get the Tool object containing all available function declarations.

    Uses FunctionDeclaration.from_callable() from the Google GenAI SDK to generate
    tool definitions from the handler functions. The result is cached on the `types`
    module so it survives Temporal sandbox re-execution of this module.
    """
    cached = getattr(types, "_tools_cache", None)
    if cached is not None:
        return cached

    # Create client to generate FunctionDeclarations
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    client = genai.Client(api_key=api_key)

    # Generate FunctionDeclarations from callables
    tools = types.Tool(
        function_declarations=[
            types.FunctionDeclaration.from_callable(
                client=client, callable=get_weather_alerts
            ),
            types.FunctionDeclaration.from_callable(
                client=client, callable=get_location_info
            ),
            types.FunctionDeclaration.from_callable(
                client=client, callable=get_ip_address
            ),
        ]
    )
    # Store on a pass-through module so the cache survives sandbox re-execution
    types._tools_cache = tools
    return tools


# =============================================================================
# Activities
# =============================================================================


@dataclass
class GeminiChatRequest:
    """Request parameters for a Gemini chat completion."""

    model: str
    system_instruction: str
    contents: list[types.Content]
    tools: list[types.Tool]


@dataclass
class GeminiChatResponse:
    """Response from a Gemini chat completion."""

    text: str | None  # The text response, if any
    function_calls: list[dict[str, Any]]  # List of function calls (name and args)
    raw_parts: list[types.Part]  # Raw parts for conversation history


@activity.defn
async def generate_content(request: GeminiChatRequest) -> GeminiChatResponse:
    """Execute a Gemini chat completion with tool support."""
    # Create the Gemini client with SDK retries disabled (Temporal handles retries)
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

    # Configure the request with automatic function calling disabled
    # (Temporal handles tool execution, not the SDK)
    config = types.GenerateContentConfig(
        system_instruction=request.system_instruction,
        tools=request.tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # Make the API call
    response = await client.aio.models.generate_content(
        model=request.model,
        contents=request.contents,
        config=config,
    )

    # Extract function calls and text from response parts
    function_calls = []
    raw_parts = []
    text_parts = []

    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            raw_parts.append(part)
            if part.function_call:
                function_calls.append(
                    {
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args) if part.function_call.args else {},
                    }
                )
            elif part.text:
                text_parts.append(part.text)

    # Only include text if there are no function calls (avoids SDK warning)
    text = "".join(text_parts) if text_parts and not function_calls else None

    return GeminiChatResponse(
        text=text,
        function_calls=function_calls,
        raw_parts=raw_parts,
    )


@activity.defn(dynamic=True)
async def dynamic_tool_activity(args: Sequence[RawValue]) -> dict:
    """Execute a tool dynamically based on the activity name.

    This activity uses Temporal's dynamic activity feature. The activity name
    (passed via execute_activity) becomes the tool name, allowing tools to be
    added/removed without changing the workflow code.

    Handles both:
    - Tools with no parameters
    - Tools with Pydantic model parameters (nested LLM output like {'request': {...}})
    """
    # The tool name comes from the activity type (how it was invoked)
    tool_name = activity.info().activity_type
    tool_args = activity.payload_converter().from_payload(args[0].payload, dict)
    activity.logger.info(f"Running dynamic tool '{tool_name}' with args: {tool_args}")

    handler = get_handler(tool_name)

    if not inspect.iscoroutinefunction(handler):
        raise TypeError("Tool handler must be async (awaitable).")

    # Inspect the handler's signature to determine how to pass arguments
    sig = inspect.signature(handler)
    params = list(sig.parameters.values())

    if len(params) == 0:
        # No parameters
        result = await handler()
    else:
        # Get the parameter name and annotation
        param = params[0]
        param_name = param.name
        ann = param.annotation

        if isinstance(ann, type) and issubclass(ann, BaseModel):
            # Handler expects a Pydantic model
            # LLM produces nested output like {'request': {'state': 'CA'}}
            # Extract the nested dict using the parameter name
            nested_args = tool_args.get(param_name, tool_args)
            result = await handler(ann(**nested_args))
        else:
            # Plain parameters - unpack dict as keyword arguments
            result = await handler(**tool_args)

    activity.logger.info(f"Tool '{tool_name}' result: {result}")
    return result


# =============================================================================
# Workflow
# =============================================================================


@workflow.defn
class AgentWorkflow:
    """Agentic loop workflow that uses Gemini for LLM calls and executes tools."""

    @workflow.run
    async def run(self, input: str) -> str:
        """Run the agentic loop until the LLM produces a final response.

        Args:
            input: The user's initial message/query.

        Returns:
            The final text response from the LLM.
        """
        # Initialize conversation history with the user's message
        contents: list[types.Content] = [
            types.Content(role="user", parts=[types.Part.from_text(text=input)])
        ]

        # Get tools (cached - initialized by worker at startup)
        tools = [get_tools()]

        # The agentic loop
        while True:
            print(80 * "=")

            # Consult the LLM
            result = await workflow.execute_activity(
                generate_content,
                GeminiChatRequest(
                    model="gemini-2.5-flash",
                    system_instruction=SYSTEM_INSTRUCTIONS,
                    contents=contents,
                    tools=tools,
                ),
                start_to_close_timeout=timedelta(seconds=60),
            )

            # Check if there are function calls to handle
            if result.function_calls:
                # Add the model's response (with function calls) to history
                contents.append(types.Content(role="model", parts=result.raw_parts))

                # Process each function call
                for function_call in result.function_calls:
                    tool_result = await self._handle_function_call(function_call)

                    # Add the function response to history
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_function_response(
                                    name=function_call["name"],
                                    response={"result": tool_result},
                                )
                            ],
                        )
                    )

            # If no function calls, we have a final response
            else:
                print(f"No tools chosen, responding with a message: {result.text}")
                return result.text

            # Uncomment the sleep to test worker crashes later:
            # await asyncio.sleep(10)

    async def _handle_function_call(self, function_call: dict) -> str:
        """Execute a tool via dynamic activity and return the result.

        Args:
            function_call: Dict containing 'name' and 'args' for the function.

        Returns:
            The string result from the tool execution.
        """
        tool_name = function_call["name"]
        tool_args = function_call.get("args", {})

        print(f"Making a tool call to {tool_name} with args: {tool_args}")

        result = await workflow.execute_activity(
            tool_name,
            tool_args,
            start_to_close_timeout=timedelta(seconds=30),
        )

        return result


# =============================================================================
# Worker
# =============================================================================


async def main():
    config = ClientConfig.load_client_connect_config()
    config.setdefault("target_host", "localhost:7233")
    client = await Client.connect(
        **config,
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue="gemini-agent-python-task-queue",
        workflows=[
            AgentWorkflow,
        ],
        activities=[
            generate_content,
            dynamic_tool_activity,
        ],
        activity_executor=ThreadPoolExecutor(max_workers=10),
    )
    await worker.run()


if __name__ == "__main__":
    load_dotenv()  # Load .env file before anything else
    get_tools()  # Populate tool cache before worker starts
    asyncio.run(main())
