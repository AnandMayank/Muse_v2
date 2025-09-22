# -*- coding: utf-8 -*-
import base64
import json
import random

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI




def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class User:
    def __init__(self, base_url: str, api_key: str):
        self.client = self.get_client(base_url, api_key)
        self.scenario = None
        self.requirement = None
        self.target_item = None

    def get_client(self, base_url, api_key):
        client = OpenAI(base_url=base_url,
                        api_key=api_key)
        return client

    def load_user(self, scenario, requirement, target_item):
        self.scenario = scenario
        self.requirement = requirement
        self.target_item = target_item

    def clear_user(self):
        self.scenario = None
        self.requirement = None
        self.target_item = None

    def chit_chat(self, conversations, recommended_item):
        content_system = "You are roleplaying as a user engaging in a conversation with a recommender system. " \
                         "In the previous turn, the system recommended a product to you. " \
                         "Your task now is to respond with a chit-chat message rather than directly accepting or rejecting the recommendation. \n" \
                         "You are given three information. " \
                         "1. The entire conversation history between you and the recommender system. " \
                         "2. The product's text and image information that was just recommended to you. " \
                         "3. Background of your purchase trip. \n" \
                         "Your chit-chat response should:   " \
                         "1. Be related to the basic/visual features of the recommended (not the product itself), but not directly address whether you want to purchase it or not. " \
                         "2. Draw from your 'personal experiences' or interests as established in the previous conversation. " \
                         "3. It can be reasonably connected with the history conversations" \
                         "4. Potentially include a brief anecdote, opinion that's tangentially related to the the product's use or the scenario. " \
                         "5. The length is limited in 3-4 sentences. " \
                         "6. Optionally, hint at your lifestyle, preferences, or future plans without explicitly connecting them to the product recommendation. " \
                         "7. Don't ask question to the Assistant. \n" \
                         "Now, based on the context of the conversation and the product just recommended to you, generate a natural chit-chat response that avoids directly addressing the recommendation. \n" \
                         "Output only the chit-chat response."
        text_information = f"Description: {recommended_item['new_description']}"
        content_user = f"Conversations: {conversations}\n; \n; Item information: {text_information}; Background:{self.scenario}\n"
        messages =[]
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        response = self.client.chat_completion(messages=messages,
            temperature=0.2,
        )
        chit_chat = response.choices[0].message.content
        return chit_chat

    def accept(self, conversations):
        content_system = "You are now playing the role of a user engaged in a conversation with a rec-assistant. " \
                         "During this conversation, the rec-assistant successfully found a product that meets your needs\n" \
                         "You will receive two information: " \
                         "1. The content of your conversations with the rec-assistant " \
                         "2. The image and detailed information of the product you finally accepted" \
                         "Based on this information, please write a concluding statement expressing your feelings about the recommendation experience and your thoughts on the final product choice. \n " \
                         "Your statement should: " \
                         "1. Ensure that you are speaking as the user. " \
                         "2. Express gratitude for the recommendation system" \
                         "3. Explain why you think this product suits you" \
                         "4. Mention specific features of the product and relate them to your personal preferences" \
                         "Please ensure your response is natural and authentic.  " \
                         "The length of your statement should be between 1-3 sentences." \
                         "Please generate your response " \
                         "Output only the personalized concluding statement. "
        content_user = []
        item_id = self.target_item["item_id"]
        description = self.target_item['description']
        features = self.target_item['features']
        title = self.target_item['title']
        image_url = f"images_main/{item_id}.jpg"
        item_texts = f"'Title: {title}; Description: {description}; some attributes: {features};'. \n"
        content_text = f"Item information: {item_texts}, Conversations: {conversations} \n"
        base64_image = encode_image(image_url)

        content_user.append({"type": "text", "text": content_text})
        content_user.append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}})

        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        response = self.client.chat_completion(messages=messages,
            temperature=0.2,
        )
        final_sentence = response.choices[0].message.content

        return final_sentence

    def find_reject_reasons(self, item):
        content_system = "You are roleplaying as a user engaged in a conversation with a rec-assistant. " \
                         "You already know exactly what item you want (your target product), but you cannot directly reveal this to the system. " \
                         "The recommender system has just suggested a product to you. \n" \
                         "Your task is to compare the features of the recommended product with your target product (including visual features) and formulate reasons to decline the recommendation. " \
                         "These reasons should be based on the differences between the two products. " \
                         "You have access to the following information: " \
                         "1. Detailed text and visual information about your target product. " \
                         "2. Detailed text and visual information about the recommended product. \n" \
                         "Your response should: " \
                         "1. Identify only one main differences(maybe on visual features) between the recommended product and your target product. " \
                         "2. Focus on the difference, using it for declining. \n" \
                         "Output the reasons for decline. \n"
        content_user = []
        content_text = ""
        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        target_item = f"TargetItem: Description: {self.target_item['description']}; Categories: {self.target_item['categories']}; Image: <Image_1>) \n"
        target_img = encode_image(f"images_main/{self.target_item['item_id']}.jpg")
        content_text += target_item
        recommended_item = f"RecommendedItem: Description: {item['description']}; Categories: {item['categories']}; Image: <Image_2>) \n"
        content_text += recommended_item
        recommended_img = encode_image(f"images_main/{item['item_id']}.jpg")
        content_user.append({"type": "text", "text": content_text})
        content_user.append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{target_img}"}})
        content_user.append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{recommended_img}"}})

        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})

        response = self.client.chat_completion(messages=messages,
            temperature=0.2,
        )
        reject_reasons = response.choices[0].message.content

        return reject_reasons

    def reject(self, conversations, item):
        reject_reason = self.find_reject_reasons(item)
        content_system = "You are roleplaying as a user engaged in a conversation with a dialogue recommender system. " \
                         "In a previous turn of the conversation (which may not be the immediately preceding turn " \
                         "due to potential chit-chat), the recommender system suggested a product to you. " \
                         "You have already found a suitable reason to decline this recommendation, and this reason " \
                         "also includes some of your own needs or preferences. " \
                         "Your task now is to respond to the recommender system, declining the suggested product " \
                         "while expressing your needs in a natural and engaging manner. " \
                         "You have access to the following information: " \
                         "1. The content of your conversation with the rec-assistant. " \
                         "2. The specific reason for declining the recommendation, which includes some of your " \
                         "needs or preferences." \
                         "Your response should:" \
                         "1. Avoid repetitive phrasing of rejections in history conversations! ." \
                         "2. Decline the recommendation based the reasons you've been provided." \
                         "3. Reference relevant parts of the conversation history to maintain context and continuity." \
                         "4. The length is limited to 1-3 sentences." \
                         "Warning: Don't mention any brand or prize! " \
                         "Now, based on the given reason for declining (which includes your needs) and the conversation history. " \
                         "Craft a response that declines the current recommendation while expressing your needs " \
                         "and keeping the conversation flowing naturally. " \
                         "Output only the response."
        content_user = []
        content_text = f"Conversations Context: {conversations}; Reject reason: {reject_reason}; "
        content_user.append({"type": "text", "text": content_text})
        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        response = self.client.chat_completion(messages=messages,
            temperature=0.2,
        )
        reject_sentence = response.choices[0].message.content

        return reject_sentence





