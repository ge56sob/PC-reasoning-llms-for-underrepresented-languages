import os
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "BlossomsAI/BloomVN-8B-Chat-Reasoning"

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
    return [{"role": "user", "content": user_prompt}]


def extract_boxed_answer(text):
    """
    Extracts the LAST answer written as \boxed{...},
    correctly handling nested braces.

    Returns:
        str | None
    """
    text = str(text)

    marker = r"\boxed{"

    # Find the last occurrence of \boxed{
    start = text.rfind(marker)

    if start == -1:
        return None

    start += len(marker)

    depth = 1
    i = start

    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    # Unmatched braces
    if depth != 0:
        return None

    return text[start:i - 1].strip()


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


def extract_between(text, start_tag, end_tag):
    text = str(text)
    start = text.find(start_tag)
    end = text.find(end_tag)

    if start == -1 or end == -1 or end <= start:
        return ""

    return text[start + len(start_tag):end].strip()


def extract_thinking_text(model_output):
    return extract_between(
        model_output,
        "<|START_THINKING|>",
        "<|END_THINKING|>",
    )


def extract_response_text(model_output):
    response = extract_between(
        model_output,
        "<|START_RESPONSE|>",
        "<|END_RESPONSE|>",
    )

    if response:
        return response

    return str(model_output).strip()


def detect_language(text, curr_lang):
    text = str(text).lower()

    vietnamese_chars = re.findall(
        r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
        r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
        r"ùúụủũưừứựửữỳýỵỷỹđ]",
        text,
    )

    vi_markers = [
        "vì", "nên", "ta", "có", "là", "vậy", "suy ra", "giả sử",
        "khi đó", "từ đó", "do đó", "bằng", "phương trình", "nghiệm",
        "tổng", "hiệu", "tích", "chia", "số", "đáp án"
    ]

    en_markers = [
        "we", "have", "therefore", "so", "thus", "since", "let",
        "then", "because", "equation", "solution", "answer",
        "sum", "product", "divide", "number", "hence"
    ]

    vi_score = len(vietnamese_chars)
    en_score = 0

    for marker in vi_markers:
        vi_score += len(re.findall(rf"\b{re.escape(marker)}\b", text))

    for marker in en_markers:
        en_score += len(re.findall(rf"\b{re.escape(marker)}\b", text))

    if vi_score == 0 and en_score == 0:
        return curr_lang

    if vi_score >= 2 and en_score >= 2:
        return "mixed"

    if vi_score > en_score:
        return "vi"

    if en_score > vi_score:
        return "en"

    return "mixed"


def response_length_stats(response_text):
    response_text = str(response_text).strip()
    words = response_text.split()

    return {
        "response_num_words": len(words),
        "response_num_chars": len(response_text),
    }


def check_coherent_reasoning(response_text):
    text = str(response_text).strip().lower()

    if len(text) < 30:
        return False

    reasoning_markers = [
        "therefore", "thus", "so", "since", "because", "we have", "let",
        "do đó", "vì", "nên", "suy ra", "giả sử", "khi đó", "từ đó"
    ]

    has_reasoning_marker = any(marker in text for marker in reasoning_markers)
    has_math_symbol = bool(re.search(r"[=+\-*/^<>]|\\frac|\\sqrt", text))

    words = text.split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    not_too_repetitive = unique_ratio > 0.25

    return has_reasoning_marker and has_math_symbol and not_too_repetitive


def analyze_output(model_output, prompt_lang):
    thinking_text = extract_thinking_text(model_output)
    response_text = extract_response_text(model_output)

    boxed_answer = extract_boxed_answer(response_text)
    has_boxed_answer = boxed_answer is not None

    reasoning_language = detect_language(thinking_text, prompt_lang)
    response_language = detect_language(response_text, prompt_lang)

    length_stats = response_length_stats(response_text)
    coherent_reasoning = check_coherent_reasoning(response_text)

    return {
        "thinking_text": thinking_text,
        "response_text": response_text,
        "has_boxed_answer": has_boxed_answer,
        "boxed_answer": boxed_answer,
        "reasoning_language": reasoning_language,
        "response_language": response_language,
        "coherent_reasoning_heuristic": coherent_reasoning,
        **length_stats,
    }
    
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
        return_tensors="pt"
    )

    input_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(input_device)

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
        skip_special_tokens=False,
    ).strip()

    return response


print("CUDA available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)
print("Tokenizer loaded.", flush=True)

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto"
)
print("Model loaded.", flush=True)

model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


for lang in LANGUAGES:
    print(f"\n================ LANGUAGE: {lang.upper()} ================", flush=True)

    dataset_dict = load_dataset("Qwen/PolyMath", lang)

    for split in SPLITS:
        print(f"\n---------- Split: {split} ----------", flush=True)

        dataset = dataset_dict[split]

        if MAX_EXAMPLES_PER_SPLIT is None:
            total = len(dataset)
        else:
            total = min(MAX_EXAMPLES_PER_SPLIT, len(dataset))

        correct_count = 0
        boxed_count = 0
        reasoning_language_consistent = 0
        response_language_consistent = 0
        response_length = 0

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

            analysis = analyze_output(model_output, prompt_lang=lang)

            predicted_answer = analysis["boxed_answer"]

            if predicted_answer is not None:
                boxed_count += 1
            else:
                predicted_answer = "NO_BOXED_ANSWER"

            normalized_pred = normalize_answer(predicted_answer)
            normalized_gold = normalize_answer(gold_answer)

            is_correct = normalized_pred == normalized_gold

            if is_correct:
                correct_count += 1

            print(f"ID: {item.get('id', i)}", flush=True)
            print(f"Question: {question}", flush=True)
            print(f"Gold: {gold_answer}", flush=True)
            print(f"Predicted: {predicted_answer}", flush=True)
            print(f"Correct: {is_correct}", flush=True)

            print("\nAnalysis:", flush=True)
            print(f"Has boxed answer: {analysis['has_boxed_answer']}", flush=True)
            print(f"Response language: {analysis['response_language']}", flush=True)
            if analysis['response_language'] == "en":
                response_language_consistent += 1
            print(
                f"Coherent reasoning heuristic: {analysis['coherent_reasoning_heuristic']}",
                flush=True,
            )
            print(f"Response words: {analysis['response_num_words']}", flush=True)
            response_length += analysis["response_num_words"]
            print(f"Response chars: {analysis['response_num_chars']}", flush=True)

            print("\nFull raw model output:", flush=True)
            print(model_output, flush=True)
            print("-" * 80, flush=True)

        accuracy = correct_count / total if total else 0
        boxed_rate = boxed_count / total if total else 0

        response_consistency_rate = (
            response_language_consistent / total if total else 0
        )

        response_length = response_length / total if total else 0

        print(f"\nResults for {lang}/{split}", flush=True)
        print(f"Correct: {correct_count}/{total}", flush=True)
        print(f"Accuracy: {accuracy:.2%}", flush=True)
        print(f"Boxed-answer rate: {boxed_rate:.2%}", flush=True)
        print(
            f"Response consistency rate: "
            f"{response_consistency_rate:.2%}",
            flush=True,
        )

        print(
            f"Average response length: {response_length:.0f} words",
            flush=True,
        )
