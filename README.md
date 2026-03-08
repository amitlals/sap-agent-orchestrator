---
title: SAP Multi-Agent Orchestrator
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.12.0
python_version: 3.12
app_file: app.py
pinned: true
license: mit
short_description: SAP Agents powered by NVIDIA NIM
tags:
  - sap
  - nvidia
  - nim
  - agents
  - abap
  - s4hana
  - multi-agent
---

# 🤖 SAP Multi-Agent Orchestrator

**Autonomous Agents × SAP × NVIDIA NIM**

A multi-agent system that intelligently routes SAP queries to specialized agents:

| Agent | Capability |
|-------|-----------|
| 💻 ABAP Code Agent | Generate, refactor, explain ABAP code with modern 7.4+ syntax |
| 📚 SAP RAG Agent | Answer SAP technical questions with built-in knowledge retrieval |
| 🔍 SQL Agent | Translate natural language → ABAP SQL and CDS view definitions |
| 🧠 Orchestrator | Smart routing — classifies queries and chains agents as needed |

## Setup

1. Create a HuggingFace Space (Gradio SDK)
2. Add `NIM_API_KEY` as a Space Secret (get one free at [build.nvidia.com](https://build.nvidia.com))
3. Push this repo to your Space

## Architecture

```
User Query → Orchestrator (LLM classify) → Agent(s) → NVIDIA NIM → Response
                                              ↑
                                    SAP Knowledge Base (in-memory RAG)
```

Built by [amitlal](https://huggingface.co/amitlal)
