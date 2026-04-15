from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from preprocess import clean_text
app = FastAPI()
Mod_path="Log_Reg_Sentimentmodel.pkl"
Vec_path="Log_Reg_Sentimentvect.pkl"
# Load model+vectorizer
model = pickle.load(open(Mod_path, "rb"))
vectorizer = pickle.load(open(Vec_path, "rb"))

class Comment(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(data: Comment):
    try:
        cleaned = clean_text(data.text)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
        label_map = {0: "Negative", 1: "Neutral",2:"Positive"}
        # return {"sentiment": int(prediction)}   # ✅ FIX
        return {"sentiment": label_map[int(prediction)]}

    except Exception as e:
        return {"error": str(e)}