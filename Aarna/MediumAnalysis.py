import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "XueZhang-bjtu/M-Thinker-7B-Iter2"

LANGUAGES = ["en", "vi"]
SPLITS = ["low", "medium", "high", "top"]
MAX_EXAMPLES_PER_SPLIT = None

MAX_NEW_TOKENS = 2000

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
            max_new_tokens=MAX_NEW_TOKENS,
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
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

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

        if MAX_EXAMPLES_PER_SPLIT is None:
            total = len(dataset)
        else:
            total = min(MAX_EXAMPLES_PER_SPLIT, len(dataset))

        correct_count = 0
        boxed_count = 0

        for i in range(total):
            item = dataset[i]

            question = item["question"]
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
                }
            )

            print(f"ID: {item.get('id', i)}")
            print(f"Question: {question}")
            print(f"Gold: {gold_answer}")
            print(f"Predicted: {predicted_answer}")
            print(f"Correct: {is_correct}")
            print("-" * 80)

        accuracy = correct_count / total if total else 0
        boxed_rate = boxed_count / total if total else 0

        print(f"\nResults for {lang}/{split}")
        print(f"Correct: {correct_count}/{total}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Boxed-answer rate: {boxed_rate:.2%}")


# ======================== ANALYSIS ========================

print("\n\n================ FINAL ANALYSIS ================")

summary = {}

for result in all_results:
    lang = result["language"]
    split = result["split"]

    key = (lang, split)

    if key not in summary:
        summary[key] = {
            "total": 0,
            "correct": 0,
            "boxed": 0,
        }

    summary[key]["total"] += 1

    if result["correct"]:
        summary[key]["correct"] += 1

    if result["predicted_answer"] != "NO_BOXED_ANSWER":
        summary[key]["boxed"] += 1


print("\nPer-language, per-split results:")

for lang in LANGUAGES:
    for split in SPLITS:
        key = (lang, split)

        if key not in summary:
            continue

        total = summary[key]["total"]
        correct = summary[key]["correct"]
        boxed = summary[key]["boxed"]

        accuracy = correct / total if total else 0
        boxed_rate = boxed / total if total else 0

        print(
            f"{lang}/{split}: "
            f"{correct}/{total} correct "
            f"({accuracy:.2%}), "
            f"boxed rate: {boxed_rate:.2%}"
        )


print("\nOverall language results:")

language_summary = {}

for lang in LANGUAGES:
    language_summary[lang] = {
        "total": 0,
        "correct": 0,
        "boxed": 0,
    }

for result in all_results:
    lang = result["language"]

    language_summary[lang]["total"] += 1

    if result["correct"]:
        language_summary[lang]["correct"] += 1

    if result["predicted_answer"] != "NO_BOXED_ANSWER":
        language_summary[lang]["boxed"] += 1


for lang in LANGUAGES:
    total = language_summary[lang]["total"]
    correct = language_summary[lang]["correct"]
    boxed = language_summary[lang]["boxed"]

    accuracy = correct / total if total else 0
    boxed_rate = boxed / total if total else 0

    print(
        f"{lang}: "
        f"{correct}/{total} correct "
        f"({accuracy:.2%}), "
        f"boxed rate: {boxed_rate:.2%}"
    )


if "en" in language_summary and "vi" in language_summary:
    en_total = language_summary["en"]["total"]
    vi_total = language_summary["vi"]["total"]

    en_accuracy = (
        language_summary["en"]["correct"] / en_total
        if en_total else 0
    )
    vi_accuracy = (
        language_summary["vi"]["correct"] / vi_total
        if vi_total else 0
    )

    gap = vi_accuracy - en_accuracy

    print("\nLanguage accuracy gap:")
    print(f"Vietnamese - English: {gap:.2%}")

    if gap > 0:
        print("Vietnamese performed better than English on this run.")
    elif gap < 0:
        print("English performed better than Vietnamese on this run.")
    else:
        print("English and Vietnamese performed equally on this run.")


print("\nIncorrect examples:")

for result in all_results:
    if result["correct"]:
        continue

    print(f"\nLanguage: {result['language']}")
    print(f"Split: {result['split']}")
    print(f"ID: {result['id']}")
    print(f"Question: {result['question']}")
    print(f"Gold: {result['gold_answer']}")
    print(f"Predicted: {result['predicted_answer']}")
    print("-" * 80)


print("\nBoxing failures:")

for result in all_results:
    if result["predicted_answer"] != "NO_BOXED_ANSWER":
        continue

    print(f"\nLanguage: {result['language']}")
    print(f"Split: {result['split']}")
    print(f"ID: {result['id']}")
    print(f"Question: {result['question']}")
    print(f"Gold: {result['gold_answer']}")
    print("-" * 80)
