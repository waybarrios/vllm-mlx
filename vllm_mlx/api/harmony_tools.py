# SPDX-License-Identifier: Apache-2.0
"""
TypeScript-style tool definition converter for Harmony/GPT-OSS models.

Harmony models expect tool definitions in TypeScript namespace format:

    namespace functions {
      // Get weather for a location
      type get_weather = (_: {
        location: string,
        unit?: "celsius" | "fahrenheit"
      }) => any;
    }

This module converts OpenAI JSON Schema tool definitions to that format.
"""

from typing import Any

# JSON Schema type to TypeScript type mapping
_TYPE_MAP = {
    "string": "string",
    "number": "number",
    "integer": "number",
    "boolean": "boolean",
    "null": "null",
    "object": "object",
}


def _convert_type(prop: dict[str, Any]) -> str:
    """
    Convert a JSON Schema property to a TypeScript type string.

    Args:
        prop: JSON Schema property definition.

    Returns:
        TypeScript type string.
    """
    # Enum: union of literal values
    if "enum" in prop:
        literals = [f'"{v}"' for v in prop["enum"]]
        return " | ".join(literals)

    schema_type = prop.get("type", "any")

    # Array with items
    if schema_type == "array":
        items = prop.get("items", {})
        item_type = _convert_type(items) if items else "any"
        return f"Array<{item_type}>"

    return _TYPE_MAP.get(schema_type, "any")


def convert_tools_to_typescript(tools: list[dict[str, Any]] | None) -> str | None:
    """
    Convert OpenAI JSON Schema tool definitions to TypeScript namespace format.

    Args:
        tools: List of tool definitions in OpenAI format, e.g.:
            [{"type": "function", "function": {"name": "...", ...}}]

    Returns:
        TypeScript namespace string, or None if no tools.
    """
    if not tools:
        return None

    functions = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name", "")
        if not name:
            continue

        description = func.get("description", "")
        parameters = func.get("parameters", {})
        properties = parameters.get("properties", {})
        required = set(parameters.get("required", []))

        # Build parameter list
        params = []
        for prop_name, prop_schema in properties.items():
            ts_type = _convert_type(prop_schema)
            optional = "?" if prop_name not in required else ""
            params.append(f"    {prop_name}{optional}: {ts_type},")

        # Build function type
        lines = []
        if description:
            lines.append(f"  // {description}")

        if params:
            params_block = "\n".join(params)
            lines.append(f"  type {name} = (_: {{\n{params_block}\n  }}) => any;")
        else:
            lines.append(f"  type {name} = () => any;")

        functions.append("\n".join(lines))

    if not functions:
        return None

    body = "\n\n".join(functions)
    return f"namespace functions {{\n{body}\n}}"
