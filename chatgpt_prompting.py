import os
import openai
import tiktoken
import pandas
import re

Tao = {
    # Value from https://platform.openai.com/account/org-settings, deleted for privacy
    "org": "",
    # Get the api key from environment variable
    "api_key": os.getenv("OPENAI_API_KEY")
}

USER = Tao
# Model choice: gpt-3.5-turbo. Most capable and lower cost.
MODEL = "gpt-3.5-turbo"
# Pricing of the model
PRICE_PER_1K_TOKENS = 0.002


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """
    Returns the number of tokens used by a list of messages. 
    Copied from section 'Counting tokens for chat API calls'. 
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


# Configuration
openai.organization = USER["org"]
openai.api_key = USER["api_key"]

# Read from dataset
df = pandas.read_excel('./data/labeled_dataset.xlsx')
# Keep only two columns
df = df[['sentence', 'type']]

# Count total number of tokens
total_num_tokens = 0
# List of all conversations. One conversation is a list of messages.
conversations = []

# Iterate over rows. One conversation for classifying one sentence.
for index, row in df.iterrows():
    messages = [
        # "gpt-3.5-turbo-0301 does not always pay strong attention to system messages."
        {"role": "system", "content": "You are a helpful assistant that detects media bias."},
        {"role": "user",
            "content": f'Classify the text from news media into one of three labels, "left", "center", or "right" in terms of US politics. It is crucial that you answer in only one word. Don\'t use any word other than the three labels. \nText:\n{row["sentence"]}\nLabel:\n'},
        # The assistant messages can be written by a dev to help give examples of desired behavior.
    ]
    total_num_tokens += num_tokens_from_messages(messages)
    conversations.append(messages)

# Estimate number of tokens and price.
price = (total_num_tokens/1000.0) * PRICE_PER_1K_TOKENS
print(f"Number of tokens: {total_num_tokens}. Price: ~${price}.")

completions = []
for idx, conversation in enumerate(conversations):
    # Create a completion
    completion = openai.ChatCompletion.create(
        model=MODEL,
        messages=conversation,
        # "Lower values like 0.2 will make it more focused and deterministic."
        # Since the task is classification, not generating creative content, use lower temperature.
        temperature=0.2,
        # Limit a response to a certain length
        max_tokens=3)
    completions.append(completion)
    print(
        "===== {:4d}/{:4d} ======".format(idx, len(conversations)))

#  Extract the assistant's reply, then transform to lower case and remove non-alphabetic characters
labels = [re.sub('[^a-z]+', '', completion['choices'][0]['message']['content'].lower())
          for completion in completions]

# By default, if the length does not match, then value is NaN
df['GPT_label'] = pandas.Series(labels)
num_matches = df[df['type'] == df['GPT_label']].shape[0]
accuracy = num_matches / df.shape[0]
print(df)
print(f"Accuracy: {accuracy}")
digits_of_accuracy = "{:.0f}".format(accuracy * 10000)[:4]

# Use tsv rather than csv to have better parsing
df.to_csv(f'GPT_1700_Acc_{digits_of_accuracy}.tsv', sep='\t')
