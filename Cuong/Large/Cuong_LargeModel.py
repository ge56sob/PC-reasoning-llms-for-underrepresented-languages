import os
import re
from datasets import load_dataset

MODEL_ID = "mradermacher/command-a-reasoning-08-2025-GGUF"

LANGUAGES = ["en", "vi"]
SPLITS = ["low", "medium", "high", "top"]

MAX_EXAMPLES_PER_SPLIT = 1
MAX_NEW_TOKENS = 2000


PROMPT_NOTES = {
    "en": r"Note: Please put the final answer in \boxed{}.",
    "vi": r"Lưu ý: Vui lòng đặt câu trả lời cuối cùng trong \boxed{}.",
}


def make_messages(question, lang):
    user_prompt = f"{question}\n\n{PROMPT_NOTES[lang]}"
    return [{"role": "user", "content": user_prompt}]


def extract_boxed_answer(text):
    marker = r"\boxed{"

    start = str(text).rfind(marker)

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

    if depth != 0:
        return None

    return text[start:i-1].strip()


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


def detect_language(text):
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
        return "unknown"

    if vi_score >= 2 and en_score >= 2:
        return "mixed"

    if vi_score > en_score:
        return "vi"

    if en_score > vi_score:
        return "en"

    return "mixed"


def reasoning_length_stats(reasoning_text):
    reasoning_text = str(reasoning_text).strip()
    words = reasoning_text.split()

    return {
        "reasoning_num_words": len(words),
        "reasoning_num_chars": len(reasoning_text),
    }


def check_coherent_reasoning(reasoning_text):
    text = str(reasoning_text).strip().lower()

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

    reasoning_language = detect_language(thinking_text)
    response_language = detect_language(response_text)
    answer_language = detect_language(boxed_answer) if boxed_answer is not None else "unknown"

    length_stats = reasoning_length_stats(thinking_text)
    coherent_reasoning = check_coherent_reasoning(thinking_text)

    answer_language_consistent = (
        answer_language == prompt_lang
        or answer_language == "unknown"
    )

    return {
        "thinking_text": thinking_text,
        "response_text": response_text,
        "has_boxed_answer": has_boxed_answer,
        "boxed_answer": boxed_answer,
        "reasoning_language": reasoning_language,
        "response_language": response_language,
        "answer_language": answer_language,
        "answer_language_consistent_with_prompt": answer_language_consistent,
        "coherent_reasoning_heuristic": coherent_reasoning,
        **length_stats,
    }


def generate_response(question, lang, llm):
    messages = make_messages(question, lang)

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0,
    )

    response = output["choices"][0]["message"]["content"]

    return response.strip()

print("Loading model...", flush=True)

from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="mradermacher/command-a-reasoning-08-2025-GGUF",
    filename="command-a-reasoning-08-2025.Q2_K.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False,
)
print("Model loaded.", flush=True)


for lang in LANGUAGES:
    print(f"\n================ LANGUAGE: {lang.upper()} ================", flush=True)

    dataset_dict = load_dataset("Qwen/PolyMath", lang)

    for split in SPLITS:
        print(f"\n---------- Split: {split} ----------", flush=True)

        dataset = dataset_dict[split]
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
                llm=llm,
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
            print(f"Reasoning language: {analysis['reasoning_language']}", flush=True)
            print(f"Response language: {analysis['response_language']}", flush=True)
            print(f"Answer language: {analysis['answer_language']}", flush=True)
            print(
                "Answer language consistent with prompt: "
                f"{analysis['answer_language_consistent_with_prompt']}",
                flush=True,
            )
            print(
                f"Coherent reasoning heuristic: {analysis['coherent_reasoning_heuristic']}",
                flush=True,
            )
            print(f"Reasoning words: {analysis['reasoning_num_words']}", flush=True)
            print(f"Reasoning chars: {analysis['reasoning_num_chars']}", flush=True)

            print("\nThinking text:", flush=True)
            print(analysis["thinking_text"], flush=True)

            print("\nResponse text:", flush=True)
            print(analysis["response_text"], flush=True)

            print("\nFull raw model output:", flush=True)
            print(model_output, flush=True)
            print("-" * 80, flush=True)

        accuracy = correct_count / total if total else 0
        boxed_rate = boxed_count / total if total else 0

        print(f"\nResults for {lang}/{split}", flush=True)
        print(f"Correct: {correct_count}/{total}", flush=True)
        print(f"Accuracy: {accuracy:.2%}", flush=True)
        print(f"Boxed-answer rate: {boxed_rate:.2%}", flush=True)
