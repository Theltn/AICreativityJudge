# LLM Story Collection — Instructions

## How to Collect Stories

You need to get **20 stories from each of 5 LLMs** (100 total).

### The Setup

The folder structure is already created:
```
data/llm_stories/
├── prompts.txt          ← The 20 prompts (for reference)
├── ChatGPT/
│   ├── 01.txt           ← Story for prompt #1
│   ├── 02.txt           ← Story for prompt #2
│   └── ...              ← through 20.txt
├── Claude/
│   ├── 01.txt
│   └── ...
├── Gemini/
├── Perplexity/
└── Copilot/
```

### Step-by-Step

1. Open `data/llm_stories/prompts.txt` to see all 20 prompts
2. Go to each LLM's web interface
3. For each prompt, paste this message:

```
Write a short story (~500 words) based on this prompt:

"[PASTE THE PROMPT HERE]"

Just write the story directly, no preamble or commentary.
```

4. Copy the story output
5. Save it as `data/llm_stories/[LLM_NAME]/[PROMPT_NUMBER].txt`
   - Example: `data/llm_stories/ChatGPT/01.txt` for ChatGPT's response to prompt #1

### Tips to Save Time

- **Open all 5 LLMs in separate tabs** at once
- **Do one prompt at a time across all 5 LLMs** (rather than all 20 on one LLM first)
- Each story should take ~30 seconds to generate, so expect ~1-2 hours total
- You can do this in batches — the scoring script will process whatever files exist

### LLM Web Interfaces

| LLM | URL |
|---|---|
| ChatGPT | https://chat.openai.com |
| Claude | https://claude.ai |
| Gemini | https://gemini.google.com |
| Perplexity | https://perplexity.ai |
| Copilot | https://copilot.microsoft.com |

### When You're Done

Run:
```bash
python scripts/score_llm_stories.py
```

This will:
1. Load all stories from the folders
2. Score each through the trained RoBERTa model
3. Generate comparison plots and a summary table
4. Save everything to `data/llm_stories/results/`
