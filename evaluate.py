import argparse
import csv
import random
from rag_query import load_model, answer_question

# simple evaluation script: reads a CSV of question/answer
# pairs and reports accuracy/bleu or other metrics.

try:
    from nltk.translate.bleu_score import sentence_bleu
except ImportError:
    sentence_bleu = None


def load_qa(path: str):
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row['question'], row['answer']))
    return pairs


def evaluate(pairs, tokenizer, model, k=5, sample=None):
    correct = 0
    total = 0
    bleu_scores = []
    if sample is not None and sample < len(pairs):
        pairs = random.sample(pairs, sample)
    for q, gold in pairs:
        total += 1
        pred = answer_question(q, tokenizer, model, k=k)
        if pred.strip().lower().startswith(gold.strip().lower()):
            correct += 1
        if sentence_bleu is not None:
            bleu_scores.append(sentence_bleu([gold.split()], pred.split()))
    acc = correct / total if total > 0 else 0.0
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else None
    return acc, avg_bleu


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the IPL RAG system.")
    parser.add_argument('qa_csv', help='CSV file containing question,answer columns')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--sample', type=int, default=None, help='random subset size')
    args = parser.parse_args()

    tokenizer, model = load_model('models/gpt-neo-ipl')
    pairs = load_qa(args.qa_csv)
    print(f'Loaded {len(pairs)} QA pairs')
    acc, avg_bleu = evaluate(pairs, tokenizer, model, k=args.k, sample=args.sample)
    print(f'Accuracy: {acc:.3f}')
    if avg_bleu is not None:
        print(f'Average BLEU: {avg_bleu:.3f}')
