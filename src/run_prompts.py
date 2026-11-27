import json
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

ALLOWED_TYPES = [
    "contradictory_results",
    "methodological_disagreement",
    "replication_failure",
    "definition_scope_mismatch",
    "effect_size_disagreement",
    "outdated_information",
]

# Strategie 1: Direkte Zero-Shot-Instruktion
PROMPT_ZERO_SHOT = """You are analyzing two scientific papers (Paper A and Paper B) to detect disagreements.
Identify the main claim on which they disagree, classify the conflict into exactly one of these categories:
- contradictory_results
- methodological_disagreement
- replication_failure
- definition_scope_mismatch
- effect_size_disagreement
- outdated_information

Return strictly valid JSON with these exact keys:
{
  "conflicting_claim": "the main claim on which the papers disagree",
  "conflict_type": "one of the six categories above",
  "explanation": "Two to four sentences explaining the disagreement."
}"""

# Strategie 2: Strukturierter Evidenz-Vergleich
PROMPT_STRUCTURED = """Compare Paper A and Paper B by explicitly contrasting their research question, methodology, dataset/scope, and empirical findings before deciding on the conflict.
Classify the disagreement into exactly one of these categories:
- contradictory_results
- methodological_disagreement
- replication_failure
- definition_scope_mismatch
- effect_size_disagreement
- outdated_information

Return strictly valid JSON with these exact keys (do not include extra keys):
{
  "conflicting_claim": "the main claim on which the papers disagree",
  "conflict_type": "one of the six categories above",
  "explanation": "Two to four sentences explaining the disagreement based on the compared evidence."
}"""

STRATEGIES = {
    "zero_shot": PROMPT_ZERO_SHOT,
    "structured": PROMPT_STRUCTURED,
}


def extract_text(paper_entry) -> str:
    if isinstance(paper_entry, str):
        return paper_entry
    if isinstance(paper_entry, dict):
        title = paper_entry.get("title", "")
        body = (
            paper_entry.get("abstract")
            or paper_entry.get("excerpt")
            or paper_entry.get("text")
            or paper_entry.get("summary")
            or ""
        )
        return f"Title: {title}\n{body}".strip()
    return str(paper_entry)


def call_groq(system_prompt: str, text_a: str, text_b: str, model: str = "openai/gpt-oss-120b") -> dict:
    user_message = f"Paper A:\n{text_a}\n\nPaper B:\n{text_b}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def main():
    with open("data/pair_excerpts.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Unterstützt sowohl eine Liste als auch ein Dictionary in pair_excerpts.json
    if isinstance(raw_data, dict):
        pairs = [{"pair_id": k, **v} if isinstance(v, dict) else {"pair_id": k, "data": v} for k, v in raw_data.items()]
    else:
        pairs = raw_data

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    for pair in pairs:
        pair_id = pair.get("pair_id") or pair.get("id")
        text_a = extract_text(pair.get("paper_a") or pair.get("paper_a_text") or "")
        text_b = extract_text(pair.get("paper_b") or pair.get("paper_b_text") or "")

        for strategy_name, system_prompt in STRATEGIES.items():
            print(f"Running {pair_id} with strategy '{strategy_name}'...")
            result = call_groq(system_prompt, text_a, text_b)
            out_file = output_dir / f"{pair_id}_{strategy_name}.json"
            with open(out_file, "w", encoding="utf-8") as out:
                json.dump(result, out, indent=2, ensure_ascii=False)

    print("Fertig! Alle Ergebnisse wurden im Ordner 'results/' gespeichert.")


if __name__ == "__main__":
    main()
