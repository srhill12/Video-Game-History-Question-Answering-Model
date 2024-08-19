
# Video Game History Question-Answering Model

This project uses the `transformers` library to build a question-answering model focused on the history of video games. The model leverages the `distilbert-base-cased-distilled-squad` model to extract relevant answers from a provided text.

## Getting Started

To run this project in Google Colab:

### 1. Install Dependencies

To start, you need to install the `transformers` library:

```python
!pip install transformers
```

### 2. Initialize the Question-Answering Pipeline

After installing the necessary libraries, you can import and initialize the pipeline using the `distilbert-base-cased-distilled-squad` model:

```python
from transformers import pipeline

# Initialize the question-answerer pipeline
question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')
```

### 3. Load and Process the Text Data

Load your text data, which in this case is a document detailing the history of video games:

```python
filepath = "Resources/video_game_history.txt"
with open(filepath) as f:
    video_game_history = f.read().replace("\n", " ")
```

### 4. Generate Questions and Extract Answers

You can generate a set of questions related to the text and use the model to find the answers:

```python
questions = [
    "When did Nintendo release its Nintendo Entertainment System in the United States?",
    "What was the first home video game?",
    "When did internet gaming start?"
]

# Example: Get the answer to a single question
question = "When did Nintendo release its Nintendo Entertainment System in the United States?"
result = question_answerer(question=question, context=video_game_history)
print(result)
```

### 5. Automate Question-Answering for Multiple Questions

You can also automate the process of answering multiple questions and store the results in a DataFrame for easy analysis:

```python
import pandas as pd

def question_answer(questions, text):
    data = []
    for question in questions:
        result = question_answerer(question=question, context=text)
        data.append([question, result['answer'], result['score'], result['start'], result['end']])
    df = pd.DataFrame(data, columns=["Question", "Answer", "Score", "Starting Position", "Ending Position"])
    return df

# Run the function and display the results
df = question_answer(questions, video_game_history)
print(df)
```

## Example Output

Here’s an example of the output from the model:

| Question                                               | Answer            | Score  | Starting Position | Ending Position |
|--------------------------------------------------------|-------------------|--------|-------------------|-----------------|
| When did Nintendo release its Nintendo Entertainment... | 1985              | 0.989  | 1576              | 1580            |
| What was the first home video game?                     | Magnavox Odyssey  | 0.774  | 419               | 435             |
| When did internet gaming start?                        | late 1990s        | 0.535  | 2392              | 2402            |

## Notes

- **Dependencies**: Ensure that all dependencies are correctly installed. The key dependency for this project is the `transformers` library from Hugging Face.
- **Text Data**: The text data used in this example is about the history of video games. You can replace this with any other text data you are interested in.
- **Questions**: Modify the list of questions to suit your specific use case.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
