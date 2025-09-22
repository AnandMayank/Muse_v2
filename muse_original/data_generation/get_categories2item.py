
import json

# 读取类型名列表
with open('categories.json', 'r') as f:
    categories = json.load(f)

# 读取item_id到categories的映射
with open('updated_item_profile.json', 'r') as f:
    item_categories = json.load(f)

# 创建新的字典，结构为 category: [item_ids]
category_items = {category: [] for category in categories}

# 遍历item_categories字典
for item_id, item_inf in item_categories.items():
    item_cats = item_inf['categories']
    for category in item_cats:
        if category in category_items:
            category_items[category].append(item_id)
# 将结果保存到新的JSON文件
with open('category2items.json', 'w') as f:
    json.dump(category_items, f, indent=4)

print("处理完成，结果已保存到 category_items.json")