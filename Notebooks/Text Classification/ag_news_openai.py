from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import json
import timeit
import time

load_dotenv()
client = OpenAI()

dataset = load_dataset('ag_news')
df_test = pd.DataFrame(dataset['test'])

res_df_path = '../../data/openai_text_classification.csv'

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_classification_of_news_article",
            "description": "Get classification of news article",
            "parameters": {
                "type": "object",
                "properties": {
                    "classification": {
                        "type": "string",
                        "description": "the classification label of the news article",
                        "enum": [
                            "world",
                            "sports",
                            "business",
                            "technology",
                            "science"
                        ]   
                    }
                },
                "required": [ "classification" ]
            }
        }
    }
]

def get_openai_classification(news_text):
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo-0125',
        messages=[
            {'role': 'system', 'content': 'You are an agent that classifies news articles into appropriate categories. Return the classification as world if no other classification is appropriate.'},
            {'role': 'user', 'content': f'{news_text}'}
        ],
        tools=tools,
        tool_choice='auto'
    )

    label = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)['classification']
    
    if label == 'world':
        return 0
    if label == 'sports':
        return 1
    if label == 'business':
        return 2
    if label == 'technology':
        return 3
    if label == 'science':
        return 3

res_list = []

for index, row in df_test[0:1000].iterrows():
    retry_count = 0
    while(True):
        try:
            print(f'=== Running Row {index} ===')
            start_time = timeit.default_timer()
            classification = get_openai_classification(row['text'])
            execution_time = timeit.default_timer() - start_time
            retry_count = 0
            break
        except Exception as e:
            print(e)
            print('*** Sleeping For Rate Limiting ***')
            time.sleep(10)
            retry_count = retry_count + 1
            if retry_count < 3:
                continue
            else:
                retry_count = 0
                classification = 0
                execution_time = 1.0
                break

    dict_to_add = dict()
    dict_to_add['index'] = index
    dict_to_add['text'] = row['text']
    dict_to_add['predicted_label'] = classification
    dict_to_add['actual_label'] = row['label']
    dict_to_add['execution_time'] = execution_time

    res_list.append(dict_to_add)

    if index % 100 == 0:
        print('*** Writing to disc ***')
        df_res = pd.DataFrame(res_list)
        df_res.to_csv(res_df_path, index=False)
    
print('Done!')