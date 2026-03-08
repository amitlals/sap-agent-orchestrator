"""
SAP Multi-Agent Orchestrator — Powered by NVIDIA NIM
=====================================================
A Gradio app hosting autonomous SAP agents:
  1. ABAP Code Agent — generates, refactors, explains ABAP code
  2. SAP RAG Agent — answers SAP technical questions with retrieval
  3. SQL Agent — translates natural language → ABAP SQL / CDS views
  4. Orchestrator — routes queries to the right agent or chains them

Deploy: huggingface.co/spaces/amitlal
"""

import os
import json
import gradio as gr
import requests
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NIM_API_KEY = os.environ.get("NIM_API_KEY", "")
NIM_BASE_URL = os.environ.get(
    "NIM_BASE_URL",
    "https://integrate.api.nvidia.com/v1"
)
NIM_MODEL = os.environ.get("NIM_MODEL", "meta/llama-3.1-70b-instruct")


# ---------------------------------------------------------------------------
# SAP Knowledge Base (lightweight in-memory RAG store)
# ---------------------------------------------------------------------------
SAP_KNOWLEDGE = {
    "tables": {
        "BKPF": "Accounting Document Header — key fields: BUKRS, BELNR, GJAHR, BLDAT, BUDAT, BLART",
        "BSEG": "Accounting Document Segment — key fields: BUKRS, BELNR, GJAHR, BUZEI, KOART, HKONT",
        "EKKO": "Purchasing Document Header — key fields: EBELN, BUKRS, BSTYP, BSART, LIFNR, EKORG",
        "EKPO": "Purchasing Document Item — key fields: EBELN, EBELP, MATNR, WERKS, MENGE, NETPR",
        "VBAK": "Sales Document Header — key fields: VBELN, AUART, VKORG, VTWEG, SPART, KUNNR",
        "VBAP": "Sales Document Item — key fields: VBELN, POSNR, MATNR, KWMENG, NETWR",
        "MARA": "General Material Data — key fields: MATNR, MTART, MBRSH, MATKL, MEINS",
        "KNA1": "Customer Master General — key fields: KUNNR, NAME1, LAND1, ORT01, STRAS",
        "LFA1": "Vendor Master General — key fields: LIFNR, NAME1, LAND1, ORT01, STRAS",
        "T001": "Company Codes — key fields: BUKRS, BUTXT, ORT01, LAND1, WAERS",
        "MARC": "Plant Data for Material — key fields: MATNR, WERKS, EKGRP, DISMM, DISPO",
        "MSEG": "Material Document Segment — key fields: MBLNR, MJAHR, ZEESSION, BWART, MATNR, WERKS, MENGE",
    },
    "modules": {
        "FI": "Financial Accounting — General Ledger, Accounts Payable/Receivable, Asset Accounting",
        "CO": "Controlling — Cost Centers, Profit Centers, Internal Orders, Product Costing",
        "MM": "Materials Management — Purchasing, Inventory, Invoice Verification",
        "SD": "Sales & Distribution — Sales Orders, Pricing, Billing, Shipping",
        "PP": "Production Planning — MRP, Shop Floor, BOM, Routing",
        "HR/HCM": "Human Capital Management — Personnel Admin, Payroll, Time Management",
        "PM": "Plant Maintenance — Maintenance Orders, Equipment, Functional Locations",
        "QM": "Quality Management — Inspection Lots, Quality Notifications, Certificates",
        "WM/EWM": "Warehouse Management — Storage Bins, Transfer Orders, Picking",
        "BTP": "Business Technology Platform — Cloud Foundry, Kyma, Integration Suite, HANA Cloud",
    },
    "rap_patterns": {
        "managed": "Managed RAP: framework handles CRUD — define behavior with 'managed implementation in class'",
        "unmanaged": "Unmanaged RAP: developer handles persistence — use when wrapping legacy BAPIs",
        "projection": "Projection Layer: exposes subset of BO for specific UI — uses 'projection' keyword in BDEF",
        "draft": "Draft Handling: enables save-as-draft — add 'with draft' to behavior definition",
    },
    "cds_annotations": {
        "@AbapCatalog": "Controls DDIC view generation, buffering, DB hints",
        "@Analytics": "Marks CDS as analytical query or cube — e.g., @Analytics.query: true",
        "@ObjectModel": "Defines associations, compositions, semantic keys for RAP",
        "@UI": "Controls Fiori Elements UI layout — lineItem, identification, selectionField, facets",
        "@Consumption.filter": "Marks fields as filter parameters in analytical queries",
        "@Semantics": "Assigns business meaning — amount, currency, quantity, unitOfMeasure",
    },
}


def search_knowledge(query: str) -> str:
    """Simple keyword-based retrieval from SAP knowledge base."""
    query_lower = query.lower()
    results = []

    for category, items in SAP_KNOWLEDGE.items():
        for key, value in items.items():
            if (key.lower() in query_lower or
                any(word in value.lower() for word in query_lower.split() if len(word) > 2)):
                results.append(f"[{category.upper()}] {key}: {value}")

    return "\n".join(results[:8]) if results else "No specific SAP knowledge found — using general knowledge."


# ---------------------------------------------------------------------------
# NVIDIA NIM API Call
# ---------------------------------------------------------------------------
def call_nim(messages: list[dict], temperature: float = 0.3, max_tokens: int = 2048) -> str:
    """Call NVIDIA NIM endpoint (OpenAI-compatible)."""
    if not NIM_API_KEY:
        return (
            "⚠️ NIM_API_KEY not set. Add it as a HF Space secret.\n\n"
            "For demo, here's a simulated response based on your query.\n"
            "Set your NVIDIA NIM API key to get real LLM-powered responses."
        )

    headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": NIM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(
            f"{NIM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ NIM API error: {str(e)}"


# ---------------------------------------------------------------------------
# Agent Definitions
# ---------------------------------------------------------------------------
class AgentType(Enum):
    ABAP_CODE = "abap_code"
    SAP_RAG = "sap_rag"
    SQL_AGENT = "sql_agent"
    ORCHESTRATOR = "orchestrator"


AGENT_SYSTEM_PROMPTS = {
    AgentType.ABAP_CODE: """You are an expert ABAP developer agent. You:
- Generate clean, modern ABAP code (7.4+ syntax with inline declarations, string templates, etc.)
- Refactor legacy ABAP to modern patterns (RAP, CDS, clean ABAP)
- Explain ABAP code clearly with line-by-line comments
- Suggest S/4HANA migration paths (BAPI → RAP, SE38 → ADT, classic reports → CDS+Fiori)
- Follow SAP clean code guidelines

Always output code in proper ABAP code blocks. Include comments explaining key decisions.
When refactoring, show BEFORE and AFTER with explanations.""",

    AgentType.SAP_RAG: """You are an SAP technical knowledge agent with deep expertise across all SAP modules.
You answer questions using the retrieved SAP context provided to you.

When answering:
- Reference specific tables, transactions, and technical details
- Distinguish between ECC and S/4HANA differences where relevant
- Include relevant CDS views, BAPIs, or function modules
- Mention SAP Notes or OSS references when applicable
- For BTP topics, reference specific services and APIs

Be precise and technical. SAP consultants and developers are your audience.""",

    AgentType.SQL_AGENT: """You are an ABAP SQL and CDS view expert agent. You translate natural language data requests
into ABAP SQL SELECT statements or CDS view definitions.

Rules:
- Use ABAP SQL syntax (not standard SQL): ASCENDING/DESCENDING instead of ASC/DESC
- Use proper SAP table names and field names
- Add appropriate WHERE clauses, JOINs, and aggregations
- For CDS views: include proper annotations (@AbapCatalog, @UI, @ObjectModel)
- Always explain the table relationships and field choices
- Suggest performance optimizations (secondary indices, buffer settings)

Output both the SQL/CDS code AND a plain-English explanation of what it does.""",

    AgentType.ORCHESTRATOR: """You are the SAP Multi-Agent Orchestrator. You analyze the user's query and determine
which specialist agent(s) should handle it. You can also chain agents for complex queries.

Available agents:
1. ABAP_CODE — for code generation, refactoring, and code explanation
2. SAP_RAG — for SAP knowledge questions, configuration, and best practices
3. SQL_AGENT — for data queries, CDS views, and ABAP SQL

Classify the query and respond with a JSON object:
{"agents": ["agent_name"], "reasoning": "why this agent", "refined_query": "optimized query for the agent"}

For complex queries, list multiple agents in execution order.
Valid agent names: ABAP_CODE, SAP_RAG, SQL_AGENT""",
}


def classify_query(query: str) -> dict:
    """Use the orchestrator to classify and route the query."""
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPTS[AgentType.ORCHESTRATOR]},
        {"role": "user", "content": query},
    ]

    result = call_nim(messages, temperature=0.1, max_tokens=300)

    # Parse JSON from response
    try:
        # Try to extract JSON from the response
        json_start = result.find("{")
        json_end = result.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            return json.loads(result[json_start:json_end])
    except json.JSONDecodeError:
        pass

    # Fallback: keyword-based routing
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["code", "abap", "class", "method", "refactor", "program", "report", "function module"]):
        return {"agents": ["ABAP_CODE"], "reasoning": "Code-related query detected", "refined_query": query}
    elif any(kw in query_lower for kw in ["select", "sql", "cds", "query", "table data", "fetch", "extract data"]):
        return {"agents": ["SQL_AGENT"], "reasoning": "Data query detected", "refined_query": query}
    else:
        return {"agents": ["SAP_RAG"], "reasoning": "General SAP knowledge query", "refined_query": query}


AGENT_TYPE_MAP = {
    "ABAP_CODE": AgentType.ABAP_CODE,
    "SAP_RAG": AgentType.SAP_RAG,
    "SQL_AGENT": AgentType.SQL_AGENT,
}


def run_agent(agent_name: str, query: str) -> tuple[str, str]:
    """Run a specific agent and return (agent_label, response)."""
    agent_type = AGENT_TYPE_MAP.get(agent_name, AgentType.SAP_RAG)

    # For RAG agent, prepend retrieved context
    context = ""
    if agent_type == AgentType.SAP_RAG:
        context = search_knowledge(query)
        query_with_context = f"Retrieved SAP Context:\n{context}\n\nUser Question: {query}"
    elif agent_type == AgentType.SQL_AGENT:
        context = search_knowledge(query)
        query_with_context = f"Available SAP Tables & Context:\n{context}\n\nUser Request: {query}"
    else:
        query_with_context = query

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPTS[agent_type]},
        {"role": "user", "content": query_with_context},
    ]

    response = call_nim(messages, temperature=0.3, max_tokens=2048)
    label = f"🤖 {agent_name} Agent"
    return label, response


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------
def orchestrate(query: str, history: list) -> str:
    """Main entry point: classify → route → execute → combine."""
    if not query.strip():
        return "Please enter a query about SAP — code, data, or knowledge questions welcome!"

    # Step 1: Classify
    classification = classify_query(query)
    agents = classification.get("agents", ["SAP_RAG"])
    reasoning = classification.get("reasoning", "")
    refined_query = classification.get("refined_query", query)

    output_parts = []
    output_parts.append(f"**🧠 Orchestrator Routing**")
    output_parts.append(f"Agents: `{'` → `'.join(agents)}` | Reason: {reasoning}")
    output_parts.append("---")

    # Step 2: Execute agents in sequence
    for agent_name in agents:
        if agent_name in AGENT_TYPE_MAP:
            label, response = run_agent(agent_name, refined_query)
            output_parts.append(f"### {label}")
            output_parts.append(response)
            output_parts.append("---")

    return "\n\n".join(output_parts)


def run_single_agent(agent_name: str, query: str) -> str:
    """Direct agent call (bypass orchestrator)."""
    if not query.strip():
        return "Please enter a query."

    context_info = search_knowledge(query)
    label, response = run_agent(agent_name, query)

    output = f"### {label}\n\n{response}"
    if context_info and "No specific" not in context_info:
        output += f"\n\n---\n**📚 Retrieved Context:**\n```\n{context_info}\n```"
    return output


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
}
.agent-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 16px;
    color: white;
    text-align: center;
}
.agent-header h1 { color: #76b900; margin: 0; font-size: 1.8em; }
.agent-header p { color: #ccc; margin: 4px 0 0 0; }
"""

HEADER_HTML = """
<div class="agent-header">
    <h1>🤖 SAP Multi-Agent Orchestrator</h1>
    <p>Autonomous Agents × SAP × NVIDIA NIM</p>
    <p style="font-size: 0.85em; color: #999;">by amitlal • Code Gen • RAG • SQL Translation • Smart Routing</p>
</div>
"""

EXAMPLE_QUERIES = [
    "Generate an ABAP class that reads purchase orders from EKKO/EKPO and calculates total spend per vendor",
    "What is the difference between managed and unmanaged RAP in S/4HANA?",
    "Show me all sales orders over $50K from last quarter — give me the ABAP SQL and CDS view",
    "Refactor this classic ABAP report to use RAP with CDS views for Fiori Elements",
    "Explain how BTP Integration Suite connects to on-premise SAP via Cloud Connector",
    "Create a CDS view with UI annotations for a Fiori Elements list report on vendor invoices",
]

with gr.Blocks(title="SAP Multi-Agent Orchestrator", css=CUSTOM_CSS) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        # ---- Tab 1: Orchestrator (Auto-Route) ----
        with gr.TabItem("🧠 Orchestrator (Auto-Route)"):
            gr.Markdown("The orchestrator analyzes your query and routes it to the best agent(s) automatically.")
            chatbot = gr.Chatbot(height=500, type="messages")
            msg = gr.Textbox(
                placeholder="Ask anything about SAP — code, data, architecture, config...",
                label="Your Query",
                lines=3,
            )
            with gr.Row():
                send_btn = gr.Button("🚀 Send", variant="primary")
                clear_btn = gr.Button("🗑️ Clear")

            gr.Examples(examples=EXAMPLE_QUERIES, inputs=msg, label="Try these →")

            def respond(message, history):
                response = orchestrate(message, history)
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response})
                return "", history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            send_btn.click(respond, [msg, chatbot], [msg, chatbot])
            clear_btn.click(lambda: ([], ""), None, [chatbot, msg])

        # ---- Tab 2: Direct Agent Access ----
        with gr.TabItem("🎯 Direct Agent Access"):
            gr.Markdown("Bypass the orchestrator — talk directly to a specific agent.")

            with gr.Row():
                agent_selector = gr.Radio(
                    choices=["ABAP_CODE", "SAP_RAG", "SQL_AGENT"],
                    value="ABAP_CODE",
                    label="Select Agent",
                )
            direct_input = gr.Textbox(
                placeholder="Enter your query for the selected agent...",
                label="Query",
                lines=3,
            )
            direct_btn = gr.Button("⚡ Run Agent", variant="primary")
            direct_output = gr.Markdown(label="Agent Response")

            direct_btn.click(
                run_single_agent,
                inputs=[agent_selector, direct_input],
                outputs=direct_output,
            )

        # ---- Tab 3: SAP Knowledge Explorer ----
        with gr.TabItem("📚 SAP Knowledge Base"):
            gr.Markdown("Browse the built-in SAP knowledge that powers the RAG agent.")

            with gr.Accordion("📊 SAP Tables", open=True):
                table_data = [[k, v] for k, v in SAP_KNOWLEDGE["tables"].items()]
                gr.Dataframe(
                    value=table_data,
                    headers=["Table", "Description & Key Fields"],
                    interactive=False,
                )

            with gr.Accordion("📦 SAP Modules", open=False):
                module_data = [[k, v] for k, v in SAP_KNOWLEDGE["modules"].items()]
                gr.Dataframe(
                    value=module_data,
                    headers=["Module", "Description"],
                    interactive=False,
                )

            with gr.Accordion("🏗️ RAP Patterns", open=False):
                rap_data = [[k, v] for k, v in SAP_KNOWLEDGE["rap_patterns"].items()]
                gr.Dataframe(
                    value=rap_data,
                    headers=["Pattern", "Description"],
                    interactive=False,
                )

            with gr.Accordion("🏷️ CDS Annotations", open=False):
                cds_data = [[k, v] for k, v in SAP_KNOWLEDGE["cds_annotations"].items()]
                gr.Dataframe(
                    value=cds_data,
                    headers=["Annotation", "Usage"],
                    interactive=False,
                )

        # ---- Tab 4: Architecture ----
        with gr.TabItem("🏛️ Architecture"):
            gr.Markdown("""
## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              🧠 Orchestrator Agent                       │
│         (Classifies & routes queries)                    │
│         Powered by NVIDIA NIM LLM                        │
└───────┬──────────────┬──────────────┬───────────────────┘
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ 💻 ABAP Code │ │ 📚 SAP RAG   │ │ 🔍 SQL Agent │
│    Agent     │ │    Agent     │ │              │
│              │ │              │ │              │
│ • Generate   │ │ • Table KB   │ │ • NL → ABAP  │
│ • Refactor   │ │ • Module KB  │ │   SQL        │
│ • Explain    │ │ • RAP KB     │ │ • NL → CDS   │
│ • Migrate    │ │ • CDS KB     │ │   views      │
└──────────────┘ └──────────────┘ └──────────────┘
        │              │              │
        ▼              ▼              ▼
┌─────────────────────────────────────────────────────────┐
│              NVIDIA NIM Inference API                     │
│           (LLaMA 3.1 70B / Mixtral / etc.)              │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| LLM Backend | NVIDIA NIM | Free tier, fast inference, OpenAI-compatible API |
| Agent Framework | Custom lightweight | No heavy deps, HF Spaces friendly, full control |
| RAG Store | In-memory dict | Zero infra, instant startup, easy to extend |
| UI | Gradio | Native HF Spaces support, chat + direct modes |
| Routing | LLM-based + keyword fallback | Smart routing with reliable fallback |

### Extending the Agents

1. **Add more knowledge**: Expand `SAP_KNOWLEDGE` dict with tables, BAPIs, transactions
2. **Add new agents**: Create new `AgentType` + system prompt + route in orchestrator
3. **Add vector RAG**: Replace keyword search with NVIDIA NeMo Retriever embeddings
4. **Add ABAP execution**: Connect to SAP via ADT API for live code validation
5. **Add memory**: Store conversation context for multi-turn agent interactions
""")

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch()
