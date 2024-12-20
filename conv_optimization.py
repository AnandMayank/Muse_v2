import json

from openai import OpenAI

import random

from tqdm import tqdm

class Rewriter:
    def __init__(self, base_url, api_key):
        self.base_url=base_url
        self.api_key=api_key
        self.client = self.get_client(base_url=base_url, api_key=api_key)
    def get_client(self, base_url, api_key):
        client = OpenAI(base_url=base_url,
                                api_key=api_key)
        return client
    def rewrite(self):
        for i in tqdm(range(1, 11111111111111)):
            with open(f'***********path_to_convs************/conv_{i}.json', 'r') as file:
                data = json.load(file)
            conversations = data['Conversations']
            new_data = data.copy()
            for conv_id in range(len(conversations)):
                conversation = conversations[conv_id]
                if random.random() < 0.2:
                    oral = "(4) You may use some colloquial expression. \n"
                else:
                    oral = "\n"
                content_system_user = "You are a helpful rewrite assistant " \
                                      "Please rewrite the following conversation from a user to a recommendation assistant.\n" \
                                      "Rules are followed: \n" \
                                      "   (1) Maintain main content of the conversation " \
                                      "   (2) Use different sentence structures and diverse words. " \
                                      "   (3) try not to increase the length of the conversation . " \
                                      f"{oral}" \
                                      "Output only the rewritten conversation!. Do not output anything else!"
                content_system_rec = "Please rewrite the following conversation from a recommendation assistant to a user.\n" \
                                     "Rules are followed: \n" \
                                     "   (1) Maintain main content of the conversation. " \
                                     "   (2) Use different sentence structures and diverse words. " \
                                     "   (3) try not to increase the length of the conversation . " \
                                    f"{oral}" \
                                     "Output only the rewritten conversation!. Do not output anything else!"
                if "Assistant" in conversation:
                    content_system_rewrite = content_system_rec
                    content_user_rewrite = f"Conversation: {conversation['Assistant']}"
                else:
                    content_system_rewrite = content_system_user
                    content_user_rewrite = f"Conversation: {conversation['User']}"

                messages_rewrite = []
                messages_rewrite.append({"role": "system", "content": content_system_rewrite})
                messages_rewrite.append({"role": "user", "content": content_user_rewrite})
                response_rewrite = self.client.chat.completions.create(
                    model='claude-3-5-haiku-latest',
                    messages=messages_rewrite,
                    temperature=0.9,
                )
                rewritten_conversation = response_rewrite.choices[0].message.content

                # if random.random() < 0.2:
                #     messages_rewrite_twice = []
                #     content_user_rewrite_twice = f"Conversation: {rewritten_conversation}"
                #     messages_rewrite.append({"role": "system", "content": content_system_rewrite})
                #     messages_rewrite.append({"role": "user", "content": content_user_rewrite_twice})
                #     response_rewrite = self.client.chat.completions.create(
                #         model='gpt-4o-mini',
                #         messages=messages_rewrite,
                #         temperature=0.8,
                #     )
                #     rewritten_conversation = response_rewrite.choices[0].message.content
                # else:
                #     pass

                content_system_check = "Given a original conversation and a rewritten conversation. " \
                                       "Please Check: " \
                                       "1. Whether the rewritten version keep the main content of the original conversation. " \
                                       "2. The rewritten conversation don't have anything else like 'Here is a written version'." \
                                       "If both conditions are met, answer 'Yes'. " \
                                       "If not, answer with 'No'. " \
                                       "Output only 'Yes' or 'No'. "
                content_user_check = f"Original conversation: {content_user_rewrite}. \n" \
                                     f"Rewritten conversation: {rewritten_conversation}"
                messages_check = []
                messages_check.append({"role": "system", "content": content_system_check})
                messages_check.append({"role": "user", "content": content_user_check})

                response_check = self.client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=messages_check,
                    temperature=0.1,
                )
                check_result = response_check.choices[0].message.content
                print(content_user_check)
                print(check_result)
                if 'Yes' in check_result:
                    if "Assistant" in conversation:
                        new_data['Conversations'][conv_id]['Assistant'] = rewritten_conversation
                    else:
                        new_data['Conversations'][conv_id]['User'] = rewritten_conversation
                else:
                    pass
            with open(f'rewrite_detail_convs/conv_{i}.json', 'w') as file:
                json.dump(new_data, file, indent=4)
from pydantic import BaseModel
class Answer(BaseModel):
    aspect_id: int
    score: float
class MathResponse(BaseModel):
    score: list[Answer]

class Reviewer:
    def __init__(self, base_url, api_key):
        self.base_url=base_url
        self.api_key=api_key
        self.client = self.get_client(base_url=base_url, api_key=api_key)
    def get_client(self, base_url, api_key):
        client = OpenAI(base_url=base_url,
                                api_key=api_key)
        return client
    def fileter(self):
        def extract_dialogues(detail_conversation):
            dialogues = []
            for conversation in detail_conversation:
                if "Assistant" in conversation:
                    dialogues.append(
                        f"Assistant:{conversation['Assistant']}"
                    )
                if "User" in conversation:
                    dialogues.append(
                        f"User:{conversation['User']}"
                    )

            return dialogues
        for i in tqdm(range(1, 111111111111111111)):
            with open(f'***********path_to_convs************/conv_{i}.json', 'r') as file:
                data = json.load(file)
            conversations = data['Conversations']
            for conv_id in range(len(conversations)):
                conversation = extract_dialogues(conversations[conv_id]['Conversations'])

                total_number = 0
                content_system = "You are a conversation score assistant. " \
                                 "Given a conversation, please score the conversation from three aspects, scoring from 0 to 2 points (using 0.05 increments): " \
                                 "Aspect 1. Content Quality" \
                                 "2 score: Rich, detailed content with specific and helpful information" \
                                 "1 score: Basic content with adequate but limited details" \
                                 "0 score: Poor or irrelevant content lacking substance" \
                                 "Aspect 2. Logical Fluency" \
                                 "2 score: Perfect logical flow with coherent reasoning and smooth transitions" \
                                 "1 score: Generally logical with minor inconsistencies" \
                                 "0 score: Significant logical breaks or disjointed responses" \
                                 "Aspect 3. User Consistency" \
                                 "2 score: Responses fully aligned with user's stated preferences and needs" \
                                 "1 score: Partially aligned with user's requirements" \
                                 "0 score: Misaligned with or ignoring user's expressed preferences. " \
                                 "Please output the score of the three aspects with no explanation."
                content_user = f"Conversation: {conversation}"
                messages = []
                messages.append({"role": "system", "content": content_system})
                messages.append({"role": "user", "content": content_user})

                try:
                    completion = self.client.beta.chat.completions.parse(
                        model="gpt-4o-mini",
                        messages=messages,
                        temperature=0.2,
                        response_format=MathResponse,
                    )
                    parse = completion.choices[0].message.parsed
                    first = parse.score
                    for fff in range(len(first)):
                        i = first[fff]
                        number = i.score
                        total_number +=number
                    if total_number < 4:
                        pass  ##filter
                    else:
                        with open(f'convs_{i}.json', 'r') as file:
                            json.dump(conversations[conv_id], file)
                except Exception as e:
                    print(e)