import base64
import json
import random

from openai import OpenAI
from faker import Faker
from pydantic import BaseModel

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

from create_item_db import ItemVector

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

fake = Faker()
ages = list(range(5, 71))

genders = ['male', 'female', 'unknown']
gender_weights = [45, 45, 10]  # 总和为100，unknown的权重为10

with open('category2items.json', 'r') as file:
    cloth_types = json.load(file)

category_weights = {category: len(items) for category, items in cloth_types.items()}

with open('updated_item_profile.json', 'r') as file:
    item_profiles = json.load(file)

weights = {
    'Worn-out': 0.1,
    'Work': 0.2,
    'Season': 0.15,
    'Trend': 0.1,
    'Sports': 0.15,
    'Gifts': 0.1,
    'Attend': 0.1,
    'Celebrate': 0.1
}
with open('reasons.json', 'r') as file:
    reasons = json.load(file)
def weighted_random_choice():
    weights = {
        'Worn-out': 0.02,
        'Work': 0.01,
        'Season': 0.01,
        'Trend': 0.03,
        'Sports': 0.2,
        'Gifts': 0.2,
        'Attend': 0.33,
        'Celebrate': 0.2
    }
    # 根据权重随机选择一个key
    keys = list(reasons.keys())
    selected_key = random.choices(keys, weights=[weights[key] for key in keys], k=1)[0]

    # 从选中的key的list中随机选择一个value
    selected_value = random.choice(reasons[selected_key])

    return selected_value

def generate_user_profile():
    selected_category = random.choices(list(category_weights.keys()),
                                       weights=list(category_weights.values()),
                                       k=1)[0]
    selected_item_id = random.choice(cloth_types[selected_category])
    selected_item = item_profiles[selected_item_id]
    while selected_item.get('new_description',-1) == -1:
        selected_item_id = random.choice(cloth_types[selected_category])
        selected_item = item_profiles[selected_item_id]
    profession = fake.job()
    reason = weighted_random_choice()
    age = random.choice(ages)
    gender = random.choices(genders, weights=gender_weights, k=1)[0]
    if gender == 'male':
        name = fake.name_male()
    elif gender == 'female':
        name = fake.name_female()
    else:
        name = fake.name()
    if age <= 15:
        profession = 'no-profession'
    selected_item['item_id'] = selected_item_id
    selected_item['images'] = f'images_main/{selected_item_id}.jpg'
    user = {'name': name, 'gender': gender, 'age': age,
            'profession': profession, 'reason': reason, 'cloth_type': selected_category,
            'target_item': selected_item}

    return user

def calculate_bleu_similarities(sentence, sentence_list):
    # 将输入句子分词
    reference = word_tokenize(sentence.lower())

    similarities = []
    for s in sentence_list:
        # 将列表中的每个句子分词
        s = s['scenario']
        candidate = word_tokenize(s.lower())

        # 计算BLEU得分
        score = sentence_bleu([reference], candidate)
        similarities.append(score)

    return similarities

    # 打印结果
db_path = "path_to_local_product_database"
data_path = "updated_item_profile.json"
model_name = "bge-m3"
client = OpenAI(base_url='base_url', api_key='api_key')
# with open('user_scenarios_7000.json', 'r') as file:
#     exist_scenarios = json.load(file)
# user_scenarios = exist_scenarios
user_scenarios = []
print('Start Loading Dataset...')
myDB = ItemVector(
            db_path=db_path,
            model_name=model_name,
            llm=None,
            data_path=data_path,
            force_create=False,
            use_multi_query=False,
        )
print('Dataset Loaded!')
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

while len(user_scenarios) < 7005:
    user = generate_user_profile()
    content_system_pre = "Given a user's basic information, a scenario, and a product's text/image information. " \
                         "Please judge: " \
                         "1. Whether the product can match the user's basici information. " \
                         "2. Whether the product can match the Scenario. " \
                         "Output format: " \
                         "1. If both criteria are satisfied: output 'Yes'. " \
                         "2. If either criterion is not met: output 'No'. " \
                         "3. No other output is permitted."
    content_system_user_pre = [{"type": "text", "text": f"1. gender: {user['gender']}, age: {user['age']}, "
                                                        f"profession:{user['profession']} \n "
                                                    f"2. Purchase Scenario: {user['reason']}"
                                                    f"3. Target item for the purchase trip: <Image> {user['target_item']['new_description']}"}]

    base64_image_target = encode_image(user['target_item']['images'])
    content_system_user_pre.append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image_target}"}})
    response_pre = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": content_system_pre},
            {"role": "user", "content": content_system_user_pre}
        ],
        temperature=0.2,
    )
    print(response_pre.choices[0].message.content)

    if 'Yes' in response_pre.choices[0].message.content:
        class MathReasoning(BaseModel):
            scenario: str
            requirement: str
        user_profile = f"'Name: {user['name']}, Gender: {user['gender']}, Age: {user['age']},Profession: {user['profession']}"
        content_system = "You are scenarios generator for a consumer's purchase motivation. \n" \
                         "Your goal is to create a scenario that could naturally lead to a online purchase, \n" \
                         "without explicitly mentioning the actual item bought. " \
                         "Remember, the scenario happened before the user's online purchase!" \
                         "You will be given three information:" \
                         "1. User information. 2. Basic reason for the backstory. 3. Information of the target item." \
                         "Use the provided user information to craft a believable and engaging narrative. \n" \
                         "The descriptions should: " \
                         "1. An upcoming event (It can be a significant life event or a minor everyday occurrence.)" \
                         "2. Include relevant contextual details such as events, emotions. " \
                         "3. Reveal the user's requirements for the product, but only towards two features!, no more!" \
                         "4. Related to the revealed requirements. " \
                         "Warning: Do not mention or describe the actual purchased item. \n " \
                         "Please generate a backstory suits the case."
        content_user = f"1. User information: f{user_profile}\n " \
                       f"2. Basic reason for this purchase trip: '{user['reason']}' \n" \
                       f"3. Target item: Cloth_type: {user['cloth_type']}; Descriptions:{user['target_item']['new_description']}"
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": content_system},
                {"role": "user", "content": content_user}
            ],
            response_format=MathReasoning,
            temperature=0.7
        )
        response = completion.choices[0].message.parsed
        sce = {'profile': user_profile, 'scenario': response.scenario,
               'requirements': response.requirement, 'target_item': user['target_item']}

        # ##########################################################################################
        # content_system_judge = "Given a scenario and requirements, which boost a user's purchse trip and the target item. " \
        #                        "Please judge whether the target item match the scenario." \
        #                        "Answer only with 'Yes' or 'No'."
        # content_user_judge = f"1. Scenario: {sce['scenario']}; Requirement: {sce['requirements']}\n" \
        #                      f"2. Target item: Cloth_type: {user['cloth_type']}; Descriptions:{user['target_item']['new_description']}, "
        # completion_judge = client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {"role": "system", "content": content_system_judge},
        #         {"role": "user", "content": content_user_judge}
        #     ],
        #     temperature=0.1
        # )
        # print(response_pre.choices[0].message.content, completion_judge.choices[0].message.content)
        # if 'No' in completion_judge.choices[0].message.content:
        #     continue
        #
        # ##########################################################################################
        if len(user_scenarios) == 0:
            user_scenarios.append(sce)
        bleu_scores = calculate_bleu_similarities(sce['scenario'], user_scenarios)
        if max(bleu_scores) > 0.15:
            print(max(bleu_scores))
            continue
        else:
            user_scenarios.append(sce)
            print(len(user_scenarios))
    else:
        continue

with open('user_profiles.json', 'w') as file:
    json.dump(user_scenarios, file, indent=4)

