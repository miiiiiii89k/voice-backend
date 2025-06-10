from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os

app = Flask(__name__)

# ✅ نموذج عربي خفيف
model_name = "akhooli/gpt2-small-arabic"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    message = data.get("message", "")
    output = generator(message, max_new_tokens=60, do_sample=True)
    reply = output[0]["generated_text"].replace(message, "").strip()
    return jsonify({"response": reply})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)