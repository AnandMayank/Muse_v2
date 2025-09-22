from conv_manager import Cmanager
from system_chat import Recsys
from user_chat import User
from tqdm import tqdm
import json
with open('user_profiles.json', 'r') as file:
    users = json.load(file)

api_base = 'api_base'
api_key = 'api_key'
db_path = "path_to_local_database"
data_path = "updated_item_profile.json"
model_name = "bge-m3"
user = User(base_url=api_base, api_key=api_key)




recsys = Recsys(db_path=db_path, data_path=data_path, model_path=model_name, base_url=api_base, api_key=api_key)
cmanager = Cmanager(user=user, recsys=recsys, base_url=api_base, api_key=api_key)

for u in tqdm(users[6999:7001]):
    try:
        cmanager.conv_process(u)
    except Exception as e:
        print('ERRRRRRRRRRRRRRRRRRRRRRRRRROORRRRRRRRRRRRRRRR')
        continue
