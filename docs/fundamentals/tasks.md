Natural Language Processing (NLP) is concerned with automatically understanding and processing input in the form of natural language. It consists of mapping the input to a desired output. There are different kind of tasks in NLP, for example text classification (assigning a text to a class based on the input) and text generation (generating the next words after an input). The latter task is what Large Language Models (LLMs) are trained to do. More on that in the [Overview](../start)

The Auto-LLM platform supports three different tasks by default:

## Sequence to Label

In the Sequence to Label task, the models take in sequences of text and output one label out of multiple possible ones for said data.

A classical example of this task is sentiment analysis of movie reviews, where the models apply either the label "positive" or the label "negative" to a movie review (the input sequence in this case).

>Input: *"That movie was very entertaining!"*
>
>Output: *Positive*

## Sequence to Sequence

In the Sequence to Sequence task, models take in a sequence of text, and output another sequence of text. This is the task most commonly associated with Large Language Models that come in the form of chatbots. Here the users input their sequences, often times a question or a prompt, and in return get a sequence in the form of an answer to their question or prompt.

>Input: "*In which City is the Louvre Museum located?*"
>
>Output: "*The Louvre Museum is located in Paris, France.*"

## Sequence to Structured Output

In the Sequence to Structured Output task, models take in a sequence of text and output structured text in response, often in the form of JSON. Structuring the data makes it easier to process the responses automatically afterwards.

>Input: *"Give me the names of 3 European Prime Ministers"*
>
>Output:

>```json
>{
>  "prime_ministers": [
>    {
>      "country": "France",
>      "prime_minister": "Sébastien Lecornu"
>    },
>    {
>      "country": "Netherlands",
>      "prime_minister": "Rob Jetten"
>    },
>    {
>      "country": "Spain",
>      "prime_minister": "Pedro Sánchez"
>    }
>  ]
>}
>```

