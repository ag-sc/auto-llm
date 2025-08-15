from auto_llm.dto.trainer_run_config import TrainerRunConfig


def flatten_schema(schema_props, schema_defs):
    flattened_schema = {}

    for key, prop_schema in schema_props.items():
        flattened_prop_schema = {}
        for p_key, p_value in prop_schema.items():
            if "$ref" in p_key:
                sub_schema = schema_defs[p_value.split("/")[-1]]

                if sub_schema.get("properties"):
                    sub_prop_schema = sub_schema["properties"]
                    sub_prop_schema_flattened = flatten_schema(
                        sub_prop_schema, schema_defs
                    )
                    sub_schema["properties"] = sub_prop_schema_flattened

                flattened_prop_schema.update(sub_schema)
            else:
                flattened_prop_schema[p_key] = p_value

        flattened_schema[key] = flattened_prop_schema

    return flattened_schema


def test_dummy():
    schema = TrainerRunConfig.model_json_schema()
    f_schema = flatten_schema(
        schema_props=schema["properties"],
        schema_defs=schema["$defs"],
    )
    ...
