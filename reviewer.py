import json

from openai import OpenAI

import random

from tqdm import tqdm

base_url = 'base_url'
api_key = 'api_key'
client = OpenAI(base_url=base_url,
                        api_key=api_key)

for i in tqdm(range(1, 7000)):
    with open(f'rewrite_detail_convs_version2/conv_{i}.json', 'r') as file:
        data = json.load(file)
    conversations = data['Conversations']
    new_data = data.copy()
    context = ""
    for conv_id in range(len(conversations)-1, len(conversations)):
        conversation = conversations[conv_id]
        # action = conversation['Action']
        # if "Assistant" in conversation:
        #     context += f"Assistant: {conversation['Assistant']}"
        # else:
        #     context += f"User: {conversation['User']}"



    content_system_check = "You are a conversation screener. " \
                           "Given a conversation between a user and a recommendation assistant. " \
                           "In the recommendation, the recommendation assistant has recommended some items to the user. " \
                           "Please check these recommendations to see if the recommended products are consistent with the context." \
                           "Output only 'Yes' or 'No'."
    content_user_check = context
    messages_check = []
    messages_check.append({"role": "system", "content": content_system_check})
    messages_check.append({"role": "user", "content": content_user_check})

    response_rewrite = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages_check,
        temperature=0.2,
    )
    rewritten_conversation = response_rewrite.choices[0].message.content
    print(rewritten_conversation)

    if 'Yes' in rewritten_conversation:
        with open(f'rewrite_detail_convs_version3/conv_{i}.json', 'w') as file:
            json.dump(new_data, file, indent=4)
    else:
        pass
