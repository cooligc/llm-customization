"""Minimal Flask web interface for the IPL RAG system."""
from flask import Flask, request, jsonify, render_template_string
from rag_query import load_model, answer_question

app = Flask(__name__)

# simple HTML template
HTML = '''
<!doctype html>
<html><head><title>IPL RAG Q&A</title></head><body>
<h1>Ask about IPL matches</h1>
<form method="post">
  <input name="question" size="80" placeholder="Type a question" />
  <input type="submit" value="Ask" />
</form>
{% if answer %}
  <h2>Answer</h2>
  <pre>{{answer}}</pre>
{% endif %}
</body></html>
'''

tokenizer, model = load_model('models/gpt-neo-ipl')

@app.route('/', methods=['GET', 'POST'])
def home():
    answer = None
    if request.method == 'POST':
        q = request.form.get('question', '')
        if q.strip():
            answer = answer_question(q, tokenizer, model)
    return render_template_string(HTML, answer=answer)

@app.route('/api/ask', methods=['POST'])
def api_ask():
    data = request.get_json(force=True)
    q = data.get('question', '')
    k = data.get('k', 5)
    if not q:
        return jsonify({'error': 'question missing'}), 400
    ans = answer_question(q, tokenizer, model, k=k)
    return jsonify({'question': q, 'answer': ans})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
