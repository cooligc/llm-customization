import argparse
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from index import retrieve


def load_model(model_path: str):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPTNeoForCausalLM.from_pretrained(model_path)
    return tokenizer, model


def answer_question(question: str, tokenizer, model, k: int = 5) -> str:
    contexts = retrieve(question, k=k)
    # build a prompt that includes the retrieved text as context
    prompt = question + "\n\n" + "\n".join([f"[context] {c}" for c in contexts]) + "\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=256, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Ask a question of the IPL GPT-Neo RAG system")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--k", type=int, default=5, help="number of retrieved passages to include")
    args = parser.parse_args()

    tokenizer, model = load_model("models/gpt-neo-ipl")
    answer = answer_question(args.question, tokenizer, model, k=args.k)
    print("\n=== answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
