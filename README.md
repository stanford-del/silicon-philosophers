# Silicon Philosophers

Code and paper source for *The Collapse of
Heterogeneity in Silicon Philosophers* by Yuanming Shi and Andreas Haupt. (ACM FAccT 2026)

## Layout

```
release/
├── README.md
├── requirements.txt
├── code/
│   ├── recompute_all_tables.py           Tables 2, 4–8, domain-per-model,
│   │                                     PCA (Table 6, Tables 19–21),
│   │                                     inline numbers
│   ├── verify_all_paper_claims.py        Table 1 + Appendix 8×8 KL/JS/r matrices
│   ├── normalize_bc_most_popular.py      Bourget & Chalmers normalization
│   │                                     (merged_*.json → final_normalized_100q/)
│   ├── generate_figure3.py               Figure 1 (human vs. Sonnet / human vs. Llama)
│   ├── generate_8panel_figure.py         Figure 10 (8-panel)
│   ├── prompt_sensitivity_batch.py       OpenAI variant inference
│   │                                     (GPT-4o, GPT-5.1)
│   ├── prompt_sensitivity_sonnet.py      Anthropic variant inference (Sonnet 4.5)
│   ├── prompt_sensitivity_fireworks.py   Fireworks variant inference (Llama/Mistral/Qwen)
│   │                                     + `analyze` subcommand that produces Table 26
│   ├── prepare_finetuning_data.py        DPO preference-pair construction
│   ├── model_eval.py                     per-philosopher LLM inference
│   │                                     (OpenAI / Anthropic / HuggingFace)
│   └── fireworks_sft_dpo.py              SFT vs. DPO comparison (Appendix tables)
├── paper/
│   └── figures/                          compiled PDFs used by the paper
└── data/                                 (NOT in git — see "Data" below)
    ├── final_normalized_100q/            canonical 277×100 matrices, 8 sources
    ├── merged_*_philosophers_normalized.json   raw responses + demographics
    ├── philosophers_with_countries.json        277 philosopher profiles
    ├── question_answer_options.json            100 survey questions
    ├── philosopher_dpo_{train,val}.jsonl       DPO preference pairs
    ├── improved_dpo_data/                      SFT + DPO Fireworks-format pairs
    └── prompt_sensitivity_batches/             variant responses for Table 26
        └── fireworks/                          Fireworks-side variant responses
```

## Dependencies

- Python 3.10 or later
- `pip install -r requirements.txt`
  (numpy, scipy, scikit-learn, matplotlib, tqdm)
- Only for regenerating raw responses from scratch: `transformers`, `peft`,
  `trl`, `torch`, plus API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `FIREWORKS_API_KEY`, `HF_TOKEN`). Not needed to reproduce anything the paper
  already reports.

## Data

We thank Dr. David Bourget for agreeing to the use of PhilPapers and PhilPeople web data. For individual-level data, please contact the authors or Dr. Bourget.

## Reproducing the paper

Two scripts produce almost every number in the paper. From `release/code/`:

```bash
python3 recompute_all_tables.py
python3 verify_all_paper_claims.py
```

These run in about two minutes on a laptop. Key values to expect:

- **Table 2** (matrix similarity to humans): Claude Sonnet 4.5 KL=0.052,
  JS=0.010, r=0.495; GPT-5.1 vs. GPT-4o r=0.951 with 95.7 % of paired cells
  exactly equal.
- **Table 6** (PCA): Human Var(6) = 71.1 %; GPT-4o 83.0 %; GPT-5.1 81.6 %;
  Claude Sonnet 4.5 72.3 %. Best PC1 alignment on Sonnet (|r| = 0.63).
- **Table 7** (question-correlation structure): Sonnet Elem r = 0.182\*\*,
  KL = 0.188, JS = 0.048.
- **Table 8** (DPO trade-off): base entropy 0.737 vs. FT 0.794; Corr KL
  0.195 → 0.176; Corr r 0.020 → 0.044\*.

### Figures

```bash
python3 generate_figure3.py
python3 generate_8panel_figure.py
```

Writes `figure1_human_vs_{sonnet,llama}.{pdf,png}` and
`figure1_8panel_bc.{pdf,png}` to `release/paper/figures/`.

### Prompt sensitivity (Table 26)

```bash
python3 prompt_sensitivity_fireworks.py analyze
```

This reads the variant responses shipped in
`release/data/prompt_sensitivity_batches/` and prints the summary across all
six non-fine-tuned models, including the two rows the paper reports:
Sonnet 4.5 (r=0.483, flip rate 5.3 %, mean shift −0.041) and Llama 3.1 8B
(r=0.464, flip rate 12.2 %, mean shift +0.039).

## Re-running inference or fine-tuning

These steps are optional — the paper's numbers and figures don't need them.
They regenerate the raw outputs and take significant compute / API budget.

| Step | Script | Requires |
|------|--------|----------|
| Per-philosopher LLM inference | `model_eval.py` | API keys and/or GPU |
| DPO preference-pair construction | `prepare_finetuning_data.py` | none (reads local survey data) |
| SFT vs. DPO comparison | `fireworks_sft_dpo.py` | `FIREWORKS_API_KEY`, `FIREWORKS_ACCOUNT_ID` |
| Prompt-sensitivity variant responses | `prompt_sensitivity_batch.py` (GPT-4o/5.1), `prompt_sensitivity_sonnet.py` (Sonnet), `prompt_sensitivity_fireworks.py` (Llama/Mistral/Qwen) | corresponding API key |

All tokens are read from environment variables (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `FIREWORKS_API_KEY`, `HF_TOKEN`). Nothing is hard-coded.

## Citation

```bibtex
@inproceedings{shi2026silicon,
  title     = {The Collapse of Heterogeneity in Silicon Philosophers},
  author    = {Shi, Yuanming and Haupt, Andreas},
  year      = {2026},
}
```


