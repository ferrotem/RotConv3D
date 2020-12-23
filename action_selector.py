#%%
from utils import file_reader, file_writer
import config as cfg
#%%
annotation = file_reader(cfg.ANNOTATION_PATH)

listen_class= []


for video in range(0,len(annotation)): #[117110,117051,117012,116946,116884,116842]:
    for selected_person in range(len(annotation[video]['p_l'])):
        action_list = annotation[video]['p_l'][selected_person]['a_l']
        action_list = list(set(action_list))
        my_lsit = [4,37]
        if all(elem in action_list  for elem in my_lsit):
        # if 37 in action_list:
            # print(video, action_list)
            listen_class.append({"video":video, "selected_person":selected_person})

            
file_writer(listen_class, "class_4_37.json")

#%%
# sd = file_reader("listen_class_2.json")
# %%
