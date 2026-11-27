# Detecting Scientific Conflicts with Prompting

Repository for **Assignment 1** (*Engineering with Generative AI*, Winter Semester 2026/27).

## Repository Structure
- `data/pair_excerpts.json` – Titles and abstracts of the three selected paper pairs.
- `gold_labels/` – Reference gold-standard JSON annotations (`gress-pair1.json` to `gress-pair3.json`).
- `papers/` – Full-text PDFs of all six papers (prepared for Assignment 2).
- `src/run_prompts.py` – Script executing the zero-shot and structured prompting strategies via the Groq API.
- `results/` – Generated JSON outputs for all paper pairs and prompting strategies.

## Setup and Reproduction

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

2. **Set your API key:**
Create a .env file in the root directory containing your Groq API key:

GROQ_API_KEY=your_api_key_here

3. **Run the prompting pipeline**

python src/run_prompts.py