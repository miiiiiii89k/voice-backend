from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    message = data.get("message", "")
    output = generator(message, max_new_tokens=100, do_sample=True)
    reply = output[0]["generated_text"].replace(message, "").strip()
    return jsonify({"response": reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)