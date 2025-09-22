# -*- coding: utf-8 -*-
import json
import random

from rate_limited_client import get_rate_limited_client
from pydantic import BaseModel


from create_local_item_database import ItemVector
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
try:
    from langchain_community.callbacks import get_openai_callback
except ImportError:
    from langchain.callbacks import get_openai_callback
import base64


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class Recsys:
    def __init__(self, db_path: str, data_path: str, model_path: str, base_url: str, api_key: str) -> None:
        self.client = self.get_client(base_url=base_url, api_key=api_key)
        self.db_path = db_path
        self.data_path = data_path
        self.model_path = model_path
        self.item_db = self.get_item_db()
        self.recommended_items = []
        self.last_query = ""
        self.prompts = self.get_prompts()

    def get_item_db(self):
        myDB = ItemVector(
            db_path=self.db_path,
            model_name=self.model_path,
            llm=None,
            data_path=self.data_path,
            force_create=True,  # Changed to True to ensure database is created
            use_multi_query=False,
        )
        return myDB

    def get_client(self, base_url, api_key):
        # Use rate-limited client instead of direct OpenAI client
        return get_rate_limited_client()

    def get_prompts(self):
        prompts = {}
        with open('prompts/chit_chat.txt', 'r') as file:
            prompts['chit_chat'] = file.read()
        with open('prompts/clarifier.txt', 'r') as file:
            prompts['clarify'] = file.read()
        with open('prompts/recommender.txt', 'r') as file:
            prompts['recommend'] = file.read()
        with open('prompts/query_generator.txt', 'r') as file:
            prompts['query'] = file.read()
        return prompts

    def clear(self):
        self.recommended_items = []
        self.last_query = ""


    def querier(self, last_query, conversations, mentioned_ids):
        requirements = self.get_requirements(conversations)
        new_query = self.clarify(requirements)
        result_item, final_query = self.once_query(last_query, new_query, mentioned_ids)

        return result_item, final_query
    def get_requirements(self, conversations):
        content_system = "You are a conversation analyzer. " \
                         "Please carefully analyze the following conversations between a user and a conversational recommendation system. \n" \
                         "Based on the conversations, identify and summarize the core needs and intentions of the user. \n" \
                         "Pay special attention to: " \
                         "1. The user's directly expressed needs or questions " \
                         "2. The user's implicit needs or areas of interest " \
                         "3. The user's emotional state and tone " \
                         "4. The user's feedback on the system's responses " \
                         "5. Any specific preferences, limitations, or criteria mentioned by the user " \
                         "Clearly summary the user's main needs for the products. " \
                         "Output only the summary."
        content_user = f"Conversation Context: {conversations}"

        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": content_system},
                {"role": "user", "content": content_user}
            ],
            temperature=0.2,
        )

        requirements = response.choices[0].message.content
        return requirements
    def clarify(self, new_query):
        content_system = "You are a professional product query clarification assistant. " \
                         "You will be given a initial query from a user for shopping products." \
                         "Your task is to transform vague or general product requests in the query into more specific, precise queries. " \
                         "This will help to more accurately match the user's actual needs. \n" \
                         "Please follow these guidelines:" \
                         "1. Analyze the user's initial query, identifying any vague or unclear parts." \
                         "2. Convert ambiguous descriptions into specific, searchable product features or categories." \
                         "3. Retain any parts of the user's query that are already clear." \
                         "4. If the user's query contains multiple aspects, clarify each aspect separately. " \
                         "5. If there is no vague part, directly return the initial query.\n" \
                         "Here are some examples: " \
                         "User Query: Waterproof bag. " \
                         "Clarified: Waterproof backpack or handbag, made of nylon or PVC material. " \
                         "Explanation: Specified bag types and common waterproof materials. \n" \
                         "Now, please clarify the user's query according to the above guidelines and examples." \
                         "Output only the clarified query"
        content_user = f"Initial query: {new_query} \n "
        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        response = self.client.chat_completion(messages=messages,
            temperature=0.2,
        )
        final_query = response.choices[0].message.content

        return final_query
    def once_query(self, last_query, new_query, mentioned_ids):
        content_system = "You are a useful query generator. " \
                         "Specifically, you are tasked to help user find suitable products in the dataset by generate specific query. \n" \
                         "You will be given two information: " \
                         "1. Old query you generated before suggesting the user's preference extracted in the history. " \
                         "2. New query analyzed from current conversations. \n" \
                         "Use these two information to generate a new query. " \
                         "Please return only the query. "
        # content_system = self.prompts['query']
        content_user = f"Old query: {last_query} \n " \
                       f"New query: {new_query} \n"
        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        response = self.client.chat_completion(messages=messages,
            temperature=0.2,
        )

        new_query = response.choices[0].message.content
        final_query = self.clarifier(new_query)
        self.item_db.retriever.search_kwargs = {"k": 8}
        results = self.item_db.search_retriever(final_query)

        filtered_results = []
        for result in results:
            if result.metadata['item_id'] in mentioned_ids:
                continue
            filtered_results.append(result)
        if len(filtered_results) == 0:
            result = None
        else:
        # results = rerank(results)
            result = random.choice(filtered_results).metadata

        return result, final_query
    def mmopen_find_outfit(self, target_item, scenario):
        class MathResponse(BaseModel):
            outfit_compatibility: bool
            scenario_appropriateness: bool

        description = target_item['new_description']

        content_system = "You are a professional fashion stylist with an eye for complementary pairings. " \
                         "I will provide you with a description of a fashion item. " \
                         "Your task is to suggest a single item that would pair well with the described piece, focusing primarily on visual compatibility. \n" \
                         "Please respond with a brief but detailed description of the suggested matching item. \n" \
                         "Your description should: " \
                         "1. Specify the type of item. " \
                         "2. Describe its key visual elements such as color, pattern, texture, and style. " \
                         "Keep your response concise, aiming for 2-3 sentences. \n" \
                         "Warning: Do not repeat or reference the original item's description!!!!! " \
                         "Focus solely on describing the matching item you're suggesting. " \
                         "Now, based on the item description I provide, please suggest a matching piece following these guidelines."

        content_user = []
        content_user.append(
            {"type": "text", "text": f"Description: {description}"})
        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2
        )
        response = completion.choices[0].message.content
        pair_item = self.item_db.search_retriever(response)[0].metadata

        if pair_item['item_id'] == target_item['item_id']:
            return None

        content_system = "As a professional fashion consultant with expertise in style matching and occasion-appropriate dressing, your task is to evaluate two pieces of information: " \
                         "1. Descriptions of two items a user intends to purchase " \
                         "2. The specific scenario or occasion for which the user is shopping. \n" \
                         "Based on this information, please provide your expert opinion on two key points: " \
                         "1. Outfit Compatibility: Assess whether the two items work well together as an outfit (only on visual features). " \
                         "Consider factors such as: " \
                         "- Color coordination - Style consistency " \
                         "2. Scenario Appropriateness: Determine if the combination is suitable for the given scenario. " \
                         "Take into account: " \
                         "- Dress code requirements (if any) - Level of formality " \
                         "- Practical considerations (weather, activities involved, etc.) " \
                         "- Cultural or social expectations \n" \
                         "For each point, provide a clear 'True' or 'False' judgment " \
                         "Please ensure your advice is practical, considerate of fashion principles, and tailored to the specific items and scenario provided."

        content_user = f"Item_1: {description}. Item_2: {pair_item['new_description']}. Scenario: {scenario}"
        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            response_format=MathResponse,
        )
        response = completion.choices[0].message.parsed
        if response.outfit_compatibility and response.scenario_appropriateness:
            return pair_item
        else:
            return None

    def chatter_chit_chat(self, conversations):
        content_system = "You are an advanced conversational recommender system engaging in a friendly chat with a user. \n" \
                         "In the previous round of conversation, you recommended an item to the user, but instead of directly accepting or rejecting the recommendation, " \
                         "the user responded with a piece of casual chit-chat related to the recommended item. \n" \
                         "Your task is to: \n" \
                         "1. Respond to the user's chit-chat, maintaining a natural flow of conversation. " \
                         "2. Utilize all previous dialogue content with this user to demonstrate an understanding of their interests and experiences. " \
                         "3. Subtly showcase your personality and humor in the chit-chat, but maintain an appropriate balance without overshadowing the conversation. " \
                         "4. Reguide the conversation back to your last round of recommendations in an appropriate way. " \
                         "Remember, your response should: \n" \
                         "1. Be relevant to the user's chit-chat topic. " \
                         "2. Show empathy and understanding and be natural and engaging. " \
                         "3. Length should be limited to 3-4 sentences. " \
                         "4. Don't ask question to the user!" \
                         "Now, based on the user's previous chit-chat statement, generate an appropriate chit-chat response. \n" \
                         "Output only the chit-chat statement."
        # content_system = self.prompts['chit_chat']
        content_user = f"History conversations: {conversations} \n "
        messages = []
        messages.append({"role": "system", "content": content_system})
        messages.append({"role": "user", "content": content_user})
        response = self.client.chat_completion(messages=messages,
            temperature=0.2,
        )
        chit_chat = response.choices[0].message.content
        return chit_chat

    def chatter_recommendation(self, conversations, recommended_item):
        content_system = "You are an advanced conversational recommender system engaged in a chat with a user." \
                         "Your task now is to recommend a product to the user. " \
                         "You have access to two key resources: " \
                         "The entire conversation history with this user up to this point. " \
                         "A detailed text description and image of the product you need to recommend. " \
                         "Your objective is to craft a product recommendation that: " \
                         "1. Avoid repeating the same start sentences in history conversations." \
                         "2. Seamlessly fits into the current conversation flow. " \
                         "3. Demonstrates understanding of the user's preferences, needs, and previous interactions. " \
                         "4. Highlights the most relevant features of the product based on what you know about the user. " \
                         "5. Incorporates relevant details from the product's text description and visual elements. " \
                         "6. Aiming for a length of 2-3 sentences. " \
                         "Output only the the introduction for the product!"
        # content_system = self.prompts['recommend']
        content_user = []
        raw_item = recommended_item
        item = raw_item
        item_id = item["item_id"]
        description = item['description']
        features = item['features']
        title = item['title']
        categories = item['categories']
        image_url = f"images_main/{item_id}.jpg"
        item_texts = f" Title: {title}; Description: {description}; Some attributes: {features}; Categories: {categories}. \n"
        content_user.append(
            {"type": "text", "text": f"Conversation History: {conversations}. \n Recommended Item: {item_texts}"}, )

        base64_image = encode_image(image_url)
        content_user.append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}})

        response = self.client.chat_completion(messages=[
                {"role": "system", "content": content_system},
                {"role": "user", "content": content_user}
            ],
            temperature=0.2,
        )
        recommendation = response.choices[0].message.content
        return recommendation

