import ultralytics
YOLO = ultralytics.YOLO
import os
import shutil

model = YOLO("yolo11s-cls.pt")  # load an official model
model = YOLO("model.pt")  # load a custom model

path = "random_files" #path to targeted directory/folder

files = os.listdir(path) 

for index, value in enumerate(files):
    
    document_path = os.path.join(path, value)   
    
    results = model(document_path)  

    conf_score = results[0].probs.data.tolist()

    for x in range(len(conf_score)):                        # Loop through probability values
        if conf_score[x] == max(conf_score):
            name = results[0].names[x]
            os.makedirs(name, exist_ok=True)
            file_extension = os.path.splitext(value)[1]     # Extract extenstion from file
            new_name = f"{name}_{value}{file_extension}"    # New document name
            new_path = os.path.join(name, new_name)         # New location of document   
            shutil.copy(document_path, name)                # Copy to new location
            os.rename(os.path.join(name, value), new_path)  # Rename the document after copying
            break  






