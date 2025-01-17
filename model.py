import pickle
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/trained_pipeline.pkl", 'rb') as f:
    model = pickle.load(f)


classes  = [
    'Arabic', 'Chinese', 'Dutch', 'English', 'Estonian', 'French',
       'Hindi', 'Indonesian', 'Japanese', 'Korean', 'Latin', 'Persian',
       'Portugese', 'Pushto', 'Romanian', 'Russian', 'Spanish', 'Swedish',
       'Tamil', 'Thai', 'Turkish', 'Urdu'
]

def predict_pipeline(text):
        text = re.sub(r'[!@#$%^&*(),:;~`?%\n"\0-9]', ' ', text)
        text = re.sub(r"[[]]", " ", text)
        text = text.lower()
        pred = model.predict([text])
        return classes[pred[0]]