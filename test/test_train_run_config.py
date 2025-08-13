from auto_llm.dto.trainer_run_config import TrainerRunConfig


def flatten_schema(schema_props, schema_defs, flattened_schema: dict):
    for key, prop_schema in schema_props.items():
        for p_key, p_value in prop_schema.items():
            if "$ref" in p_key:
                value = schema_defs[p_value.split("/")[-1]]
                flattened_schema[key] = flatten_schema(
                    schema_props=value["properties"],
                    schema_defs=schema_defs,
                    flattened_schema=flattened_schema,
                )
    return flattened_schema


def test_dummy():
    schema = TrainerRunConfig.model_json_schema()
    f_schema = flatten_schema(
        schema_props=schema["properties"],
        schema_defs=schema["$defs"],
        flattened_schema={},
    )
    ...
