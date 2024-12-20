import base64
import json
import os

import requests
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm


# model = "https://huggingface.co/OpenGVLab/InternVL2-40B"
# def generate_product_description(text, image_path):
#     # Disable torch init
#     image = load_image(image_path)
#     pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
#     text_prompt = "You are an expert in generating product image descriptions. " \
#                   "You will receive two types of information: the title of the product, and the product image. \n" \
#                   "Please generate a description of the product's visual information. " \
#                   "Follow three steps: " \
#                   "1.Identify the part of the product in the image. 2. Focus on the product itself. 3. Describe" \
#                   f"Item Title: {text}"
#     response = pipe((text_prompt, image))
#     print(response.text)

# class MathReasoning(BaseModel):
#     Description: str
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
client = OpenAI(base_url='base_url', api_key='key')
def generate_product_description(text, image_path):
    content_system = "You are a item feature extractor. " \
                     "Given the information: \n" \
                     "1. The Image of the item. " \
                     "2. The Text information of the item" \
                     "You need to follow the steps below: \n" \
                     "1. Identify the part of the product in the image. " \
                     "2. Focus on the product itself and ignore other objects in the image. " \
                     "3. Generate a detail description of the product including all basic features and visual features. \n" \
                     "Output the only the description of the item."
    content_user = []
    text_pre = f"Text information:{text},"
    content_user.append({"type": "text", "text": f"{text_pre}"}, )
    base64_image = encode_image(image_path)
    content_user.append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}})
    messages = []
    messages.append({"role": "system", "content": content_system})
    messages.append({"role": "user", "content": content_user})
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,
    )
    return completion.choices[0].message.content

def download_image(url, save_path):
    # 发送 GET 请求到指定的 URL
    response = requests.get(url)
    if os.path.exists(save_path):
        return save_path
    # 确保请求成功
    if response.status_code == 200:
        # 获取文件名
        file_name = os.path.basename(url)

        # 组合保存路径
        full_path = save_path

        # 以二进制写模式打开文件，并写入内容
        with open(full_path, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to download image. Status code: {response.status_code}")
    return save_path

with open('item_profile_small.json') as file:
    data = json.load(file)

new_descriptions = {}
count = 1
t = 1
for id, item in tqdm(data.items()):
    count+=1
    text_information = f"Title:{item['title']}; Description: {item['description']}; " \
                       f"Feature: {item['features']}; Categories: {item['categories']}"
    try:
        save_path = f"images_main/{id}.jpg"
    except Exception as e:
        continue
#

    try:
        new_description = generate_product_description(text_information, save_path)
        new_descriptions[id] = new_description
        if count % 500 == 0:
            with open(f'new_descriptions_{t}.json', 'w') as file:
                json.dump(new_descriptions, file)
            t += 1
            new_descriptions = {}
    except Exception as e:
        continue
with open('new_descriptions_final.json', 'w') as file:
    json.dump(new_descriptions, file)
merged_data = {}
for i in range(1, 11):
    filename = f"new_descriptions_{i}.json"

    # 检查文件是否存在
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            # 加载JSON数据
            data = json.load(file)
            # 更新合并后的字典
            merged_data.update(data)
    else:
        print(f"文件 {filename} 不存在")

# 将合并后的数据写入新的JSON文件
with open("new_descriptions.json", 'w', encoding='utf-8') as outfile:
    json.dump(merged_data, outfile, ensure_ascii=False, indent=4)

print("合并完成,结果保存在 merged_descriptions.json 文件中")


# 读取两个 JSON 文件
with open('new_descriptions.json', 'r', encoding='utf-8') as f1:
    new_descriptions = json.load(f1)

with open('item_profile.json', 'r', encoding='utf-8') as f2:
    items = json.load(f2)

# 将第一个文件中的 new_description 添加到第二个文件中
for item_id, new_description in new_descriptions.items():
    if item_id in items:
        items[item_id]['new_description'] = new_description  # 将 new_description 融入到第二个文件中

# 保存修改后的文件
with open('updated_item_profile.json', 'w', encoding='utf-8') as f_out:
    json.dump(items, f_out, ensure_ascii=False, indent=4)

