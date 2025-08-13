import gradio as gr

from auto_llm.dto.trainer_run_config import TrainerRunConfig
from collections import deque


def flatten_json_schema(schema: dict) -> dict:
    """
    Iteratively flattens a JSON schema by resolving nested objects and $refs.

    Args:
        schema (dict): The original nested JSON schema.

    Returns:
        dict: A new dictionary representing the flattened schema, where keys
              are dot-separated paths to the original properties.
    """
    # This will hold the final flattened schema
    flattened_schema = {}

    # We use a deque as a queue for iterative processing
    # Each item is a tuple: (prefix_key, current_schema_fragment)
    # The prefix_key tracks the path from the root.
    queue = deque([("", schema)])

    # Extract the schema definitions for easy lookup
    schema_defs = schema.get("$defs", {})

    while queue:
        prefix, current_schema = queue.popleft()

        # Check for a reference and resolve it if it exists
        if "$ref" in current_schema:
            ref_path = current_schema["$ref"].split("/")
            # Look up the definition in the provided schema_defs
            current_schema = schema_defs.get(ref_path[-1], {})

        # Process the properties of the current schema fragment
        properties = current_schema.get("properties", {})
        required_fields = current_schema.get("required", [])

        for key, prop_schema in properties.items():
            # Construct the new, flattened key
            new_key = f"{prefix}.{key}" if prefix else key

            # Check if the property is a nested object or has another reference
            if prop_schema.get("type") == "object" or "$ref" in prop_schema:
                # Add the nested object to the queue for processing
                queue.append((new_key, prop_schema))
            else:
                # This is a terminal property, add it to our flattened schema
                # We also add a 'required' flag for form generation
                prop_schema_copy = prop_schema.copy()
                prop_schema_copy["required"] = key in required_fields
                flattened_schema[new_key] = prop_schema_copy

    return flattened_schema


def build_gradio_ui_from_schema(schema: dict, schema_defs: dict, parent_key: str = ""):
    """
    Recursively builds a Gradio UI from a JSON schema.

    Args:
        schema (dict): The JSON schema fragment to process.
        schema_defs (dict): The global `$defs` section of the schema for resolving references.
        parent_key (str): The key of the parent object for nested structures.

    Returns:
        A list of Gradio components.
    """
    components = {}

    # Check for a reference to a defined schema object
    if "$ref" in schema:
        ref_path = schema["$ref"].split("/")
        # Get the schema definition from the global $defs
        schema = schema_defs[ref_path[-1]]

    # Process properties of the current schema object
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    for key, prop_schema in properties.items():
        # Get the field title and a descriptive label
        field_title = prop_schema.get("title", key).replace("_", " ").title()
        is_required = key in required_fields
        label = f"{field_title}{' (Required)' if is_required else ''}"

        common_features = {
            "label": label,
            "value": prop_schema.get("default"),
            "interactive": True,
            "info": prop_schema.get("description"),
        }

        # Check for a nested object or a reference
        if prop_schema.get("type") == "object" or "$ref" in prop_schema:
            with gr.Accordion(f"{field_title}", open=False):
                # Recursively build the UI for the nested object
                nested_components = build_gradio_ui_from_schema(
                    prop_schema, schema_defs, parent_key=key
                )
                components[key] = nested_components

        # Check for enum types (dropdowns)
        elif "enum" in prop_schema:
            print("enum", label)
            components[key] = gr.Dropdown(
                label=label,
                choices=prop_schema["enum"],
                value=prop_schema["enum"][0],
                interactive=True,
            )

        # Handle arrays, specifically for `target_modules`
        elif prop_schema.get("type") == "array":
            # For arrays of strings, we'll use a Textbox and expect comma-separated values
            default_value = (
                ", ".join(prop_schema["default"])
                if isinstance(prop_schema["default"], list)
                else ""
            )
            components[key] = gr.Textbox(
                label=f"{label} (comma-separated values)",
                value=default_value,
                interactive=True,
            )

        # Handle boolean types (checkboxes)
        elif prop_schema.get("type") == "boolean":
            components[key] = gr.Checkbox(**common_features)

        # Handle integer and number types
        elif "anyOf" in prop_schema:
            # This handles cases like `logging_steps` which can be integer or number
            components[key] = gr.Number(label=label, value=prop_schema.get("default"))
        elif prop_schema.get("type") in ["integer", "number"]:
            components[key] = gr.Number(label=label, value=prop_schema.get("default"))

        # Handle string types (textboxes)
        elif prop_schema.get("type") == "string":
            components[key] = gr.Textbox(**common_features)

    return components


def load_run_config_json(*components):
    return components


with gr.Blocks() as demo:
    # build trainer run config

    with gr.Tab("Define"):
        gr.Markdown("Define the trainer run configuration here")
        schema = TrainerRunConfig.model_json_schema()
        schema_defs = schema["$defs"]
        components = build_gradio_ui_from_schema(schema, schema_defs)
        btn = gr.Button("Submit")
    with gr.Tab("Validate"):
        gr.Markdown("## Validation")
        run_config_json = gr.JSON()
    with gr.Tab("Train"):
        gr.Markdown("## Training")

    # btn.click(
    #     fn=load_run_config_json,
    #     inputs=components_state,
    #     outputs=[run_config_json],
    # )


if __name__ == "__main__":
    demo.launch()
