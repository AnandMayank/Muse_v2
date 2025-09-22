import json

from openai import OpenAI
from tqdm import tqdm

with open('updated_item_profile.json', 'r') as file:
    data = json.load(file)
cate = {}
for item_id, metadata in data.items():
    categories = metadata['categories']
    for i in categories:
        if i in cate:
            cate[i] = cate[i]+1
        else:
            cate[i] = 1
client = OpenAI(base_url='base_url', api_key='api_key')
cate_li = []
for cate, num in tqdm(cate.items()):
    if num < 5:
        continue
    content_system = "You are an AI assistant specialized in identifying categories of clothing, footwear, and accessories. " \
                     "I will give you a label, and you need to determine if this label belongs to one of the following three categories: \n" \
                     "1. Types of clothing (e.g., jacket, down coat, T-shirt, etc. " \
                     "2. Types of footwear (e.g., high heels, sneakers, sandals, etc. " \
                     "3. Types of accessories/decorative items (e.g., necklace, handbag, hat, etc.) \n" \
                     "If the given label belongs to any of the above three categories, please answer 'Yes'. " \
                     "If it does not belong, please answer 'No'. " \
                     "Please only answer 'Yes' or 'No' without any additional explanation. " \
                     "For example: \n" \
                     "Input: jacket " \
                     "Output: Yes " \
                     "Input: high heels " \
                     "Output: Yes " \
                     "Input: necklace " \
                     "Output: Yes " \
                     "Input: clothing " \
                     "Output: No \n " \
                     "Output only 'Yes' or 'No'. "

    content_user = [{"type": "text", "text": f"Tag: {cate}"}]

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": content_system},
            {"role": "user", "content": content_user}
        ],
        temperature=0.2,
    )
    if 'Yes' in response.choices[0].message.content:
        cate_li.append(cate)
with open('categories.json', 'w') as file:
    json.dump(cate_li, file)
