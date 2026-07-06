"""Dummy model that immediately calls the done tool. Useful for testing sandbox and scorer."""

from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo


@modelapi(name="done_caller")
class DoneCallerAPI(ModelAPI):
    def __init__(
        self,
        model_name: str = "default",
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
    ):
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, api_key_vars=api_key_vars, config=config)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # Find the done tool
        done_tool = next((t for t in tools if t.name == "done"), None)

        if done_tool:
            message = ChatMessageAssistant(
                content="Work complete.",
                tool_calls=[
                    ToolCall(
                        id="call_done",
                        function="done",
                        arguments={},
                        type="function",
                    )
                ],
            )
            stop_reason = "tool_calls"
        else:
            message = ChatMessageAssistant(content="No done tool found.")
            stop_reason = "stop"

        return ModelOutput(
            model="done_caller",
            choices=[ChatCompletionChoice(message=message, stop_reason=stop_reason)],
            usage=ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        )
