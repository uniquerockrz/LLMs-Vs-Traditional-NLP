from dotenv import load_dotenv
import os
import google.generativeai as genai
import google.ai.generativelanguage as glm
import textwrap
from datasets import load_dataset
import pandas as pd
import timeit
import time
import re

load_dotenv()

dataset = load_dataset('stanfordnlp/imdb')

df_test = pd.DataFrame(dataset['test'])

genai.configure(api_key=os.environ['GOOGLE_KEY'])
model = genai.GenerativeModel(model_name='gemini-1.0-pro')

res_df_path = '../../data/gemini_sentiment.csv'
res_list = []

for i in range(0, 25000, 10):
    retry_count = 0
    df_sub = df_test.loc[i:((i+10)-1)]
    
    while(True):
        try:
            print(f'=== Running Rows {i}:{i+10} ===')
            start_time = timeit.default_timer()
            reviews = '\n'.join([review for review in df_sub['text']])
            result = model.generate_content(f"""
                I will be providing upto 10 movie reviews. For each review, find the sentiment and tag it into either positive or negative.\n
                Return an array of the sentiments in the order of the reviews provided, separated by commas.\n
                {reviews}
            """)
            execution_time = timeit.default_timer() - start_time
            if re.search(r'(Negative,?|Positive,?){1,10}', result.text, re.IGNORECASE) == None:
                print('*** Repeating because regex didn\'t match ***')
                continue
            else:
                sentiments = [sentiment.lower().strip() for sentiment in result.text.split(',')]
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