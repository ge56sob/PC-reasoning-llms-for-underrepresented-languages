import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "tencent/HY-MT1.5-1.8B"

LANGUAGES = ["en", "vi"]
SPLITS = ["low", "medium", "high", "top"]
MAX_EXAMPLES_PER_SPLIT = None

PROMPT_NOTES = {
    "en": r"Note: Please put the final answer in \boxed{}.",
    "vi": r"Lưu ý: Vui lòng đặt câu trả lời cuối cùng trong \boxed{}.",
}


def make_messages(question, lang):
    user_prompt = f"{question}\n\n{PROMPT_NOTES[lang]}"

    return [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]


def extract_boxed_answer(text):
    """
    Extracts the last answer written as \\boxed{...}.
    Returns None if no boxed answer exists.
    """
    text = str(text)

    matches = re.findall(r"\\boxed\{([^{}]*)\}", text)

    if len(matches) == 0:
        return None

    return matches[-1].strip()


def normalize_answer(text):
    if text is None:
        return ""

    text = str(text).strip().lower()

    text = text.replace("$", "")
    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    text = text.replace("{", "")
    text = text.replace("}", "")
    text = text.replace(",", "")

    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".。")

    return text

def detect_language_heuristic(text):
    text = text.lower()

    vi_markers = ["và", "là", "không", "có", "cho", "một", "các", "trong", "để", "với"]
    vi_score = sum(1 for w in vi_markers if w in text)

    en_words = ["the", "is", "are", "therefore", "solution", "answer", "step"]
    en_score = sum(1 for w in en_words if w in text)

    if vi_score > en_score and vi_score > 0:
        return "vi"
    if en_score > 0:
        return "en"
    return "unknown"


def coherence_score(text):
    text = text.lower()
    score = 0.0

    if any(k in text for k in ["therefore", "thus", "hence", "step", "1.", "2."]):
        score += 0.4

    if any(k in text for k in ["=", "+", "-", "\\frac", "because"]):
        score += 0.3

    words = text.split()
    if len(words) > 0:
        rep = len(words) / len(set(words))
        if rep < 1.3:
            score += 0.3

    return min(score, 1.0)


def reasoning_length(text):
    idx = text.rfind("\\boxed")
    if idx == -1:
        return len(text.split())
    return len(text[:idx].split())

def backtranslate_vi_to_en(question, tokenizer, model):
    prompt = f"""
Translate this Vietnamese math problem into clear English.
Do NOT solve it. Only translate.

Question:
{question}
"""

    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    out = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(out, skip_special_tokens=True).strip()

def generate_response(question, lang, tokenizer, model):
    messages = make_messages(question, lang)

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return response


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


all_results = []

for lang in LANGUAGES:
    print(f"\n================ LANGUAGE: {lang.upper()} ================")

    dataset_dict = load_dataset("Qwen/PolyMath", lang)

    for split in SPLITS:
        print(f"\n---------- Split: {split} ----------")

        dataset = dataset_dict[split]
        total = len(dataset)

        correct_count = 0
        boxed_count = 0
        language_match_count = 0
        coherence_sum = 0.0
        reasoning_sum = 0

        for i in range(total):
            item = dataset[i]

            question = item["question"]

            if lang == "vi":
                question = backtranslate_vi_to_en(question, tokenizer, model)
            gold_answer = str(item["answer"]).strip()

            model_output = generate_response(
                question=question,
                lang=lang,
                tokenizer=tokenizer,
                model=model,
            )

            predicted_answer = extract_boxed_answer(model_output)

            if predicted_answer is not None:
                boxed_count += 1
            else:
                predicted_answer = "NO_BOXED_ANSWER"

            normalized_pred = normalize_answer(predicted_answer)
            normalized_gold = normalize_answer(gold_answer)

            is_correct = normalized_pred == normalized_gold
            lang_detected = detect_language_heuristic(model_output)
            lang_match = (lang_detected == lang)

            coherence = coherence_score(model_output)
            reason_len = reasoning_length(model_output)

            if lang_match:
                language_match_count += 1

            coherence_sum += coherence
            reasoning_sum += reason_len

            if is_correct:
                correct_count += 1

            all_results.append(
           {
            "language": lang,
            "split": split,
            "id": item.get("id", i),
            "question": question,
            "gold_answer": gold_answer,
            "model_output": model_output,
            "predicted_answer": predicted_answer,
            "correct": is_correct,

            # NEW METRICS
            "detected_language": lang_detected,
            "language_match": lang_match,
            "coherence_score": coherence,
            "reasoning_length": reason_len,
          }
        )

            print(f"ID: {item.get('id', i)}")
            print(f"Question: {question}")
            print(f"Gold: {gold_answer}")
            print(f"Predicted: {predicted_answer}")
            print(f"Correct: {is_correct}")
            print(f"Detected language: {lang_detected}")
            print(f"Language match: {lang_match}")
            print(f"Coherence score: {coherence:.2f}")
            print(f"Reasoning length: {reason_len} words")
            print("-" * 80)

        accuracy = correct_count / total if total else 0
        boxed_rate = boxed_count / total if total else 0
        language_rate = language_match_count / total if total else 0
        avg_coherence = coherence_sum / total if total else 0
        avg_reasoning = reasoning_sum / total if total else 0

        print(f"\nResults for {lang}/{split}")
        print(f"Correct: {correct_count}/{total}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Boxed-answer rate: {boxed_rate:.2%}")
        print(f"Language match rate: {language_rate:.2%}")
        print(f"Average coherence score: {avg_coherence:.2f}")
        print(f"Average reasoning length: {avg_reasoning:.1f} words")
