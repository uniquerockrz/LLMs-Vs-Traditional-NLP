from dotenv import load_dotenv
import os
import google.generativeai as genai
import google.ai.generativelanguage as glm
import textwrap
from datasets import load_dataset
import pandas as pd
import timeit
import time

load_dotenv()

dataset = load_dataset('ag_news')

df_test = pd.DataFrame(dataset['test'])

genai.configure(api_key=os.environ['GOOGLE_KEY'])

res_df_path = '../../data/gemini_text_classification.csv'

classification = glm.Schema(
    type = glm.Type.OBJECT,
    properties = {
        'classification_label':  glm.Schema(type=glm.Type.STRING),
    },
    required=['classification_label']
)

classify_news_article = glm.FunctionDeclaration(
    name="classify_news_article",
    description=textwrap.dedent("""\
        Classify the news article into one of the four categories: world, sports, business, technology
        """),
    parameters=glm.Schema(
        type=glm.Type.OBJECT,
        properties = {
            'classification_label': classification
        }
    )
)

model = genai.GenerativeModel(model_name='gemini-1.0-pro', tools = [classify_news_article])

def get_gemini_classification(news_text):
    result = model.generate_content(f"""
    Please classify the provided news article into one of the following categories: world, sports, business, technology. Do not return null as a classification label.

        {news_text}
    """)
    
    fc_result = result.candidates[0].content.parts[0].function_call
    label = type(fc_result).to_dict(fc_result)['args']['classification_label']['classification_label']
    if label == 'world':
        return 0
    if label == 'sports':
        return 1
    if label == 'business':
        return 2
    if label == 'technology':
        return 3

res_list = []

for index, row in df_test.iterrows():
    retry_count = 0
    while(True):
        try:
            print(f'=== Running Row {index} ===')
            start_time = timeit.default_timer()
            classification = get_gemini_classification(row['text'])
            execution_time = timeit.default_timer() - start_time
            retry_count = 0
            break
        except Exception as e:
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