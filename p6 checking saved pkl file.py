
import pickle, os

score_folder = r"C:\Rabiul\1. PhD Research\8. Fall 2024\Lung Project\Coding\5. Coding for Publication\Code Outputs\p23 dataset variation1, mal_3 benign"
all_input_files = os.listdir(score_folder)

input_files = [filename for filename in all_input_files if filename.endswith('.pkl')]

all_scores = []
for filename in input_files:
    print(filename)
    with open(f'{score_folder}/{filename}', 'rb') as file:
        a_score = pickle.load(file)
        break
    
        

pickle_file_path = r"C:\Rabiul\1. PhD Research\10. Summer 2025\1. Research 2025\3. Collaboration for VLMs\Nodule Classification\Codes\code outputs\c10 img pred 955 nod embed\c10 ResNet50_nodule_embeddings.pkl"
with open(pickle_file_path, 'rb') as file:
    a_score = pickle.load(file)

for key, value in a_score.items():
    print(key)