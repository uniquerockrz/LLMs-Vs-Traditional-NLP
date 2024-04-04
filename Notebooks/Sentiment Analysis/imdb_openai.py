from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI
import pandas as pd
import json
import pandas as pd
import timeit
import time

load_dotenv()
client = OpenAI()

dataset = load_dataset('stanfordnlp/imdb')
df_test = pd.DataFrame(dataset['test'])

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_sentiment_of_texts",
            "description": "get sentiments of a list of texts",
            "parameters": {
                "type": "object",
                "properties": {
                    "sentiments_list": {
                        "type": "array",
                        "description": "a list of sentiments derived from a list of texts",
                        "items": {
                            "type": "object",
                            "description": "the sentiment of a single text from the list of texts",
                            "properties": {
                                "sentiment": {
                                    "type": "string",
                                    "enum": [
                                        "positive",
                                        "negative",
                                    ],
                                    "description": "The sentiment of the text, positive or negative"
                                }
                            },
                            "required": [
                                "sentiment"
                            ]
                        }
                    }
                }
            }
        }
    }
]

res_df_path = '../../data/openai_sentiment.csv'
res_list = []

for i in range(0, 1000, 10):
    retry_count = 0
    df_sub = df_test.loc[i:((i+10)-1)]
    
    while(True):
        try:
            print(f'=== Running Rows {i}:{i+10} ===')
            start_time = timeit.default_timer()
            reviews = '\n'.join(['"""' + review + '"""' for review in df_test.loc[0:10].text])
            
            completion = client.chat.completions.create(
                model='gpt-3.5-turbo-0125',
                messages=[
                    {
                        'role': 'system', 
                        'content': 'You are an agent that takes in multiple sentences separated by triple quotes and newlines, and returns the sentiment, either positive or negative, of each of them. Ignore the HTML tags and escape sequences and just consider the text.'
                    },
                    {
                        'role': 'user', 
                        'content': f'{reviews}'
                    }
                ],
                tools=tools,
                tool_choice="auto"
            )

            execution_time = timeit.default_timer() - start_time
            sentiments = json.loads(completion.choices[0].message.content)['sentiments']
            print(json.loads(completion.choices[0].message.tool_calls[0].function.arguments))

            if len(sentiments) != len(df_sub):
                print('*** Repeating because of length mismatch ***')
                retry_count = retry_count + 1
                if retry_count < 3:
                    continue
                else:
                    retry_count = 0
                    sentiment_nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    execution_time = 1.0
                    break
            else:
                sentiment_nums = []
                for sentiment in sentiments:
                    if sentiment == 'negative':
                        sentiment_nums.append(0)
                    else:
                        sentiment_nums.append(1)
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
                sentiment_nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                execution_time = 1.0
                break
    
    df = pd.DataFrame(
        index=df_sub.index,
        data={
            'text': df_sub['text'],
            'predicted_labels': sentiment_nums,
            'actual_labels': df_sub['label'],
            'execution_time': [execution_time/10] * 10
        }
    )

    res_as_dict = df.to_dict('records')
    res_list = res_list + res_as_dict
    
    if i % 100 == 0:
        print('*** Writing to disc ***')
        df_res = pd.DataFrame(res_list)
        df_res.to_csv(res_df_path, index=False)

print('Done!')