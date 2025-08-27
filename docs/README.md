<h1 align="center">
    AutoLLM
</h1>

<p align="center">
    <strong>Train and Evaluate your LMs effortlessly. 😊</strong>
</p>


# 📋 Step 1: Data Ingestion

### ``TaskDataBuilder``
- As a user, you ingest the data for the task you want to achieve with the help of an LLM.
- For example:

```python
# for PICO task
input_text = """
Title : A randomized trial in the investigation of anxiety and depression in patients with coronavirus disease 2019 ( COVID - 19 ) . METHODS : Sixty - five COVID - 19 patients were randomly enrolled into this study . Anxiety and depression among participants were measured through the completion of anonymous Chinese - language Zung self - rating anxiety scale and self - rating depression scale questionnaires . Data were analyzed using independent samples t - tests , Mann - Whitney U - tests , and ?2 tests .
"""
output_text = """
{
  "P": [
      'patients with coronavirus disease 2019 ( COVID - 19 )', 
      'Sixty - five COVID - 19 patients'
  ],
  "I": [],
  "C": [],
  "O": [
      'Anxiety', 
      'depression', 
      'anonymous Chinese - language Zung self - rating anxiety scale',
      'self - rating depression scale questionnaires'
  ],
}
"""
}
```

### ```TrainerDataBuilder```

- The ingested task specific data is further processed and converted to the format of the specific trainer you choose.
- Here you can add a couple of more features:
  - ``model_name``: which pre-trained model? Parameter Efficient FT or Full-Weights FT?
  - ``instruction``: prompt template for input and output text
  - ``few-shots``: how many? how to sample them? etc.
  - ...
- For example:
```python
# for PICO task with Supervised Fine-Tuning (SFT) Trainer - Non-Conversational Dataset
prompt = """\
Given the text "Text", extract the PICO tags in the JSON format "Format". Do not modify the sentences.
Format:
\```json
{
  "P": ["value for P"],
  "I": ["value for I"],
  "C": ["value for C"],
  "O": ["value for O"],
}
\```
Text: Title : A randomized trial in the investigation of anxiety and depression in patients with coronavirus disease 2019 ( COVID - 19 ) . METHODS : Sixty - five COVID - 19 patients were randomly enrolled into this study . Anxiety and depression among participants were measured through the completion of anonymous Chinese - language Zung self - rating anxiety scale and self - rating depression scale questionnaires . Data were analyzed using independent samples t - tests , Mann - Whitney U - tests , and ?2 tests .

PICO tags according to the format:
"""

completions = """\
\```json
{
  "P": [
      'patients with coronavirus disease 2019 ( COVID - 19 )', 
      'Sixty - five COVID - 19 patients'
  ],
  "I": [],
  "C": [],
  "O": [
      'Anxiety', 
      'depression', 
      'anonymous Chinese - language Zung self - rating anxiety scale',
      'self - rating depression scale questionnaires'
  ],
}
\```
"""
```

# ⚖️ Step 2: Define Evaluation Metrics

- Here, you should define which metrics would suit best for evaluating the task.
- For example:
```python
# for PICO task
metrics = [
    f1_score(),
    fuzzy_match()
]
```

# 🔄 Step 3: Train and Evaluate

- With the ``Evaluator``, you can evaluate the performance of pre-trained LLMs on the task you defined, against the evaluation metrics you configured.
- With the ``Trainer``, you can train different LLMs of your choice.
- ``Evaluator`` then evaluates the trained models and compare the results against the baseline results.