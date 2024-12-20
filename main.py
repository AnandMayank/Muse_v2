# This is a sample Python script.
from conv_manager import Cmanager
from system_chat import Recsys
from user_chat import User
import json
from tqdm import tqdm
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


with open('user_scenarios_7005.json', 'r') as file:
    users = json.load(file)

api_base = 'api_base'
api_key = 'api_key '
db_path = "path_to_local_item_database"
data_path = "path_to_item_json_file"
model_name = "path_to_bge-m3"
user = User(base_url=api_base, api_key=api_key)
recsys = Recsys(db_path=db_path, data_path=data_path, model_path=model_name, base_url=api_base, api_key=api_key)
cmanager = Cmanager(user=user, recsys=recsys, base_url=api_base, api_key=api_key)

for u in tqdm(users[1:9999]):
    try:
        cmanager.conv_process(u)
    except Exception as e:
        print('ERRRRRRRRRRRRRRRRRRRRRRRRRROORRRRRRRRRRRRRRRR')
        continue
