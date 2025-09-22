# -*- coding: utf-8 -*-
import base64
import json
import random

from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from user_chat import User
from system_chat import Recsys

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class Cmanager:
    def __init__(self, user: User, recsys: Recsys, base_url: str, api_key: str):
        self.user = user
        self.recsys = recsys
        self.conversations = []
        self.last_query = ""
        self.actions = []
        self.client = self.get_client(base_url, api_key)
        self.mentioned_items = []
        self.mentioned_ids = []
        self.action_conv = []
        self.max_round = 4
        self.current_round = 1
        self.conv_num = 6999
    def get_client(self, base_url, api_key):
        client = OpenAI(base_url=base_url,
                        api_key=api_key)
        return client

    def action_control(self, last_action=None):
        """
        """
        if self.current_round == self.max_round:
            return 'recommend'  # 对话结束

        # 根据轮次设置chit-chat的概率
        chit_chat_probabilities = {
            2: 0.3,
            3: 0.2,
            4: 0
        }

        # 如果连续两轮都是chit-chat，则这轮必须是推荐轮
        if last_action == 'chit-chat':
            return 'recommend'

        # 否则，根据概率决定
        if random.random() < chit_chat_probabilities[self.current_round]:
            return 'chit-chat'
        else:
            return 'recommend'

    def prepare_conv(self):
        self.conversations = []
        self.mentioned_items = []
        self.mentioned_ids = []
        self.current_round = 1
        self.actions = []
        self.action_conv = []
        self.last_query = ""

    def conv_process(self, user):
        total_loader = {}

        profile = user['profile']
        scenario = user['scenario']
        requirement = user['requirements']
        target_item = user['target_item']

        total_loader['Persona'] = profile
        total_loader['Scenario'] = scenario
        total_loader['Target_item'] = target_item['item_id']

        print('Prepared for conversation!')
        self.prepare_conv()
        self.user.clear_user()
        if random.random() < 0.2 and self.open_dialogue_multimodal(target_item, scenario):
            self.user.load_user(scenario, requirement + ', Buy for suit', target_item)
            self.max_round = 3
            print("Image Query Openning！")
        else:
            self.open_dialogue_text(scenario, requirement)
            self.user.load_user(scenario, requirement, target_item)
            self.max_round = 4
            print("Chit-chat Openning！")
        print('Start conversation!')
        finish_flag = self.one_round_conv()
        while not finish_flag:
            self.current_round += 1
            finish_flag = self.one_round_conv()
            print(self.current_round, self.actions[-1])
        print('Finish conversation!')

        total_loader['Mentioned_items'] = self.mentioned_ids
        total_loader['Conversations'] = self.action_conv

        with open(f'convs/conv_{self.conv_num}.json', 'w') as file:
            json.dump(self.conversations, file, indent=4)

        with open(f'detail_convs/conv_{self.conv_num}.json', 'w') as file:
            json.dump(total_loader, file, indent=4)
        self.conv_num+=1


    def open_dialogue_multimodal(self, target_item, scenario):
        #### STEP 1
        bought_item = self.recsys.mmopen_find_outfit(target_item, scenario)
        if bought_item == None:
            return False
        target_item = target_item
        class Round(BaseModel):
            System: str
            User: str

        class MathResponse(BaseModel):
            First_round: Round
            Second_round: Round

        content_system = "You will be given two pieces of information: " \
                    "1. Description and image of an item the user has previously purchased." \
                    "2. Description and image of the target item the user is currently interested in buying. " \
                    "Your task is to generate conversations between the user and the conversational Recommender.  \n" \
                    "The conversation should follow these guidelines: " \
                    "1. The dialogue begins with the recommendation system greeting the user. " \
                    "2. The conversation should last for 3 rounds between the system and the user. " \
                    "3. The dialogue should consist of chitchat related to a given items.  " \
                    "4. No product recommendations should be made in this conversation. " \
                    "5. Identify key attributes of the target item that the user is looking for, without explicitly naming the type of item. " \
                    "6. Explain how these attributes would match or complement the previously purchased item. " \
                    "Please generate the conversations, clearly indicating which part is spoken by the system and which by the user . " \
                    "Ensure the dialogue feels natural and flows logically based on the given context and needs." \
                    "Remember to keep the conversation focused on chit-chat related to the shopping context and user's needs, without making specific product recommendations." \
                    "Output format: " \
                    "Round1: System: ['Hi, What can I do for you!']; User: [Talk about the previous item. Show that the user wants to find a suitable outfit with it. (length: 2-3 sentences)]; " \
                    "Round2: System: [Suggest about a visual feature based on the Target item as you don't know the target item.  (length: 2-3 sentences)]]; User: [Reply, mention 1-2 visual features based on the target item (length: 2-3 sentences)]"
                    # "Round3: System: [Reply, express empathic, ask what scenario the suit for. (Length: 2 sentences)]; User: [Reply, talk about the scenario (length: 1-2 sentences)]"
        content_user = []
        content_text = {"type": "text",
                     "text": f"Bought Item's description: {bought_item['categories']}, image: <Image_1>, \n"
                             f"Target Item's description: {target_item['categories']}, image: <Image_2> \n "}
        content_user.append(content_text)

        image_url_bought = f"images_main/{bought_item['item_id']}.jpg"
        base64_image_bought = encode_image(image_url_bought)
        content_user.append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image_bought}"}})

        image_url_target = f"images_main/{target_item['item_id']}.jpg"
        base64_image_target = encode_image(image_url_target)
        content_user.append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image_target}"}})

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": content_system},
                {"role": "user", "content": content_user}
            ],
            temperature=0.2,
            response_format=MathResponse,
        )
        parse = completion.choices[0].message.parsed

        first = parse.First_round
        second = parse.Second_round

        self.conversations.append({'Assistant': first.System})
        self.conversations.append({'User': first.User})
        self.conversations.append({'Assistant': second.System})
        self.conversations.append({'User': second.User})

        self.action_conv.append({'Assistant': first.System, 'Action': 'chit-chat', 'Mentioned_item': [], 'Image': []})
        self.action_conv.append({'User': first.User, 'Action': 'chit-chat', 'Mentioned_item': [bought_item['item_id']], 'Image': []})
        self.action_conv.append({'Assistant': second.System, 'Action': 'chit-chat', 'Mentioned_item': [], 'Image': []})
        self.action_conv.append({'User': second.User, 'Action': 'chit-chat', 'Mentioned_item': [], 'Image': []})

        #### STEP 2
        content_system = "You will be given two pieces of information: " \
                         "1. Conversations happened between a user and a conversational recommender Assistant" \
                         "Your task is to continue the conversations as a conversational recommender Assistant. \n" \
                         "The conversation should follow these guidelines: " \
                         "1. Express your appreciation for the outfit mentioned in the above conversations. " \
                         "2. Ask the user what occasion this outfit will be worn on. " \
                         "3. The Length is limited in 2-3 sentences. " \
                         "Ensure the dialogue feels natural and flows logically based on the given conversations." \
                         "Remember to keep the conversation focused on chit-chat without making specific product recommendations.\n " \
                         "For example: 'It sounds like the color and style of the outfit you want are very suitable. " \
                         "What scenario will you wear them for? Knowing the occasion will allow me to consider more details and recommend good item to you.' " \
                         "Only output the conversation of this round!"
        content_user = []
        content_text = {"type": "text", "text": f"Conversations: {self.conversations}"}
        content_user.append(content_text)

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": content_system},
                {"role": "user", "content": content_user}
            ],
            temperature=0.2,
        )
        third_sys = completion.choices[0].message.content

        #### STEP_3
        content_system = "You will be given two pieces of information: " \
                         "1. Conversations happened between a user and a conversational recommender Assistant. " \
                         "2. The scenario behind the user's purchase. " \
                         "Last round, the assistant has asked a question about what scenario the the suit (mentioned in above conversations) for. " \
                         "Your task is to continue the conversations as the user. \n" \
                         "The conversation should follow these guidelines: " \
                         "1. Answer the assistant based on the scenario. " \
                         "2. Express your expectations and mention one additional requirements for target item based on the content of the scenario. " \
                         "3. The Length is limited in 2-3 sentences. " \
                         "Ensure the dialogue feels natural and flows logically based on the given conversations." \
                         "Remember to keep the conversation focused on chit-chat\n " \
                         "Example: 'Emm, I plan to wear this outfit to my granddaughter's graduation. " \
                         "So I expect the outfit is formal and can also make me look younger. " \
                         "Also, since the graduation will be a long one, I want the pants to be loose and breathable.' \n" \
                         "Only output the conversation of this round!"
        content_user = []
        content_text = {"type": "text", "text": f"Conversations: {self.conversations}; Scenario: {scenario}"}
        content_user.append(content_text)

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": content_system},
                {"role": "user", "content": content_user}
            ],
            temperature=0.2,
        )

        third_user = completion.choices[0].message.content

        self.conversations.append({'Assistant': third_sys})
        self.conversations.append({'User': third_user})

        self.action_conv.append({'Assistant': third_sys, 'Action': 'chit-chat', 'Mentioned_item': [], 'Image': []})
        self.action_conv.append({'User': third_user, 'Action': 'chit-chat', 'Mentioned_item': [], 'Image': []})
        self.mentioned_ids.append(bought_item['item_id'])
        return True

    def open_dialogue_text(self, background, requirement):
        class Round(BaseModel):
            System: str
            User: str

        class MathResponse(BaseModel):
            First_round: Round
            Second_round: Round

        content_system = "Given context:" \
                         "1. Background of a user's purchase trip." \
                         "2. Requirement of the user." \
                         "Generate conversations between a user and a conversational recommendation system. " \
                         "The conversation should follow these guidelines: " \
                         "1. The dialogue begins with the recommendation system greeting the user. " \
                         "2. The conversation should last for 2 rounds between the system and the user. " \
                         "3. The dialogue should consist of chitchat related to a given shopping context and the user's surface-level needs. " \
                         "4. No product recommendations should be made in this conversation. " \
                         "Please generate the conversation, clearly indicating which part is spoken by the system and which by the user . " \
                         "Warning: The system won't know the context unless the user has mentioned!" \
                         "Ensure the dialogue feels natural and flows logically based on the given context and needs." \
                         "Remember to keep the conversation focused on chit-chat related to the shopping context and user's needs, without making specific product recommendations." \
                         "Output format: " \
                         "Round1: System: ['Hi, What can I do for you!']; User: [Reply, mention some information of the context  (length: 2-3 sentences)]; " \
                         "Round2: System: [Reply, express empathic and ask for more(length: 2-3 sentences)]]; User: [Reply, mention other information/requirement of the context (length: 2-3 sentences)]"
        content_user = f"Background: {background}; Requirement: {requirement}"
        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=MathResponse,
            temperature=0.2,
        )
        parse = completion.choices[0].message.parsed
        first = parse.First_round
        second = parse.Second_round
        self.conversations.append({'Assistant': first.System})
        self.conversations.append({'User': first.User})
        self.conversations.append({'Assistant': second.System})
        self.conversations.append({'User': second.User})
        self.action_conv.append({'Assistant': first.System, 'Action': 'chit-chat', 'Mentioned_item': [], 'Image': []})
        self.action_conv.append({'User': first.User, 'Action': 'chit-chat', 'Mentioned_item': [], 'Image': []})
        self.action_conv.append({'Assistant': second.System, 'Action': 'chit-chat', 'Mentioned_item': [], 'Image': []})
        self.action_conv.append({'User': second.User, 'Action': 'chit-chat', 'Mentioned_item': [], 'Image': []})


    def one_round_conv(self):
        if self.current_round == 1:
            # 推荐一个商品.
            result_item, last_query = self.recsys.once_query("", self.conversations, self.mentioned_ids)
            self.mentioned_ids.append(result_item['item_id'])
            self.mentioned_items.append(result_item)
            sys_conv = self.recsys.chatter_recommendation(self.conversations, result_item)
            self.conversations.append({'Assistant': sys_conv})
            ##########logger
            self.action_conv.append({'Assistant': sys_conv, 'Action': 'recommend',
                                     'Mentioned_item': [result_item['item_id']],
                                     'Image': [f"images_main/{result_item['item_id']}"]})
            self.actions.append('recommend')
            ##########logger
            return False
        else:
            if self.actions[-1] == 'recommend' and self.mentioned_items[-1]['item_id'] == self.user.target_item['item_id']:
                acc_sentence = self.user.accept(self.conversations)

                self.conversations.append({'User': acc_sentence})
                ##########logger
                self.action_conv.append({'User': acc_sentence, 'Action': 'accept',
                                           'Mentioned_item': [], 'Image': []})
                ##########logger
                return True

            current_action = self.action_control(self.actions[-1])
            self.actions.append(current_action)

            if current_action == 'chit-chat':
                chitchat_user = self.user.chit_chat(self.conversations, self.mentioned_items[-1])
                self.conversations.append({'User': chitchat_user})
                chitchat_sys = self.recsys.chatter_chit_chat(self.conversations)
                self.conversations.append({'Assistant': chitchat_sys})

                self.action_conv.append({'User': chitchat_user, 'Action': 'chit-chat',
                                         'Mentioned_item': [], 'Image': []})
                self.action_conv.append({'Assistant': chitchat_sys, 'Action': 'chit-chat',
                                         'Mentioned_item': [], 'Image': []})

                return False
            else:
                #推荐
                reject_user = self.user.reject(self.conversations, self.mentioned_items[-1])
                self.conversations.append({'User': reject_user})

                self.action_conv.append({'User': reject_user, 'Action': 'reject',
                                         'Mentioned_item': [], 'Image': []})

                #判断是否到达了最后一轮
                if self.current_round == self.max_round:
                    result_item = self.user.target_item
                    recommend_sys = self.recsys.chatter_recommendation(self.conversations, result_item)
                    self.conversations.append({'Assistant': recommend_sys})
                    self.mentioned_items.append(result_item)
                    self.mentioned_ids.append(result_item['item_id'])
                    acc_sentence = self.user.accept(self.conversations)
                    self.conversations.append({'User': acc_sentence})

                    self.action_conv.append({'Assistant': recommend_sys, 'Action': 'recommend',
                                               'Mentioned_item': [result_item['item_id']],
                                               'Image': [f"images_main/{result_item['item_id']}"]})
                    self.action_conv.append({'User': acc_sentence, 'Action': 'accept',
                                             'Mentioned_item': [], 'Image': []})

                    return True
                else:
                    result_item, final_query = self.recsys.querier(self.last_query, self.conversations, self.mentioned_ids)
                    if result_item is None:
                        result_item = self.user.target_item
                    self.last_query = final_query
                    self.mentioned_items.append(result_item)
                    self.mentioned_ids.append(result_item['item_id'])
                    recommend_sys = self.recsys.chatter_recommendation(self.conversations, result_item)
                    self.conversations.append({'Assistant': recommend_sys})

                    self.action_conv.append({'Assistant': recommend_sys, 'Action': 'recommend',
                                             'Mentioned_item': [result_item['item_id']],
                                             'Image': [f"images_main/{result_item['item_id']}"]})

                    return False


