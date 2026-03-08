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
# Gradio UI — NVIDIA-inspired dark theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ---- Global Dark Theme ---- */
.gradio-container {
    max-width: 1200px !important;
    background: #0d0d0d !important;
}
.dark .gradio-container { background: #0d0d0d !important; }

/* ---- Header Banner ---- */
.nv-header {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 40%, #1c2b10 100%);
    border: 1px solid #2a2a2a;
    border-left: 4px solid #76b900;
    padding: 28px 32px;
    border-radius: 8px;
    margin-bottom: 20px;
}
.nv-header h1 {
    color: #76b900;
    margin: 0;
    font-size: 1.9em;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.nv-header .nv-sub {
    color: #a0a0a0;
    margin: 6px 0 0 0;
    font-size: 1em;
    font-weight: 400;
}
.nv-header .nv-meta {
    color: #666;
    font-size: 0.82em;
    margin-top: 8px;
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}
.nv-header .nv-badge {
    background: #1a2610;
    color: #76b900;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 0.78em;
    font-weight: 600;
    border: 1px solid #2d3f1a;
}

/* ---- Agent Cards ---- */
.nv-agents-row {
    display: flex;
    gap: 12px;
    margin: 12px 0 20px 0;
}
.nv-agent-card {
    flex: 1;
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 16px;
    transition: border-color 0.2s;
}
.nv-agent-card:hover { border-color: #76b900; }
.nv-agent-card h3 {
    color: #76b900;
    margin: 0 0 6px 0;
    font-size: 0.95em;
}
.nv-agent-card p {
    color: #888;
    margin: 0;
    font-size: 0.82em;
    line-height: 1.4;
}

/* ---- Inputs focus ---- */
textarea:focus {
    border-color: #76b900 !important;
    box-shadow: 0 0 0 1px #76b900 !important;
}

/* ---- Status bar ---- */
.nv-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    margin-bottom: 12px;
    font-size: 0.82em;
    color: #888;
}
.nv-dot {
    width: 8px; height: 8px;
    background: #76b900;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 6px #76b900;
}

/* ---- Footer ---- */
.nv-footer {
    text-align: center;
    padding: 16px;
    color: #444;
    font-size: 0.78em;
    border-top: 1px solid #1a1a1a;
    margin-top: 20px;
}
.nv-footer a { color: #76b900; text-decoration: none; }
"""

HEADER_HTML = """
<div class="nv-header">
    <h1>SAP Multi-Agent Orchestrator</h1>
    <p class="nv-sub">Autonomous AI Agents for SAP — Powered by NVIDIA NIM</p>
    <div class="nv-meta">
        <span class="nv-badge">NVIDIA NIM</span>
        <span class="nv-badge">LLaMA 3.1 70B</span>
        <span class="nv-badge">Multi-Agent</span>
        <span class="nv-badge">RAG</span>
        <span>by amitlal</span>
    </div>
</div>
"""

AGENTS_HTML = """
<div class="nv-agents-row">
    <div class="nv-agent-card">
        <h3>ABAP Code Agent</h3>
        <p>Generate, refactor & explain modern ABAP 7.4+ code. RAP, CDS, clean ABAP, S/4HANA migration.</p>
    </div>
    <div class="nv-agent-card">
        <h3>SAP RAG Agent</h3>
        <p>Deep SAP knowledge with retrieval. Tables, modules, BAPIs, transactions, BTP services.</p>
    </div>
    <div class="nv-agent-card">
        <h3>SQL Agent</h3>
        <p>Natural language to ABAP SQL & CDS views. Annotations, JOINs, performance optimization.</p>
    </div>
</div>
"""

STATUS_HTML = """
<div class="nv-status">
    <span class="nv-dot"></span>
    <span>NIM Inference Endpoint Active</span>
    <span style="margin-left: auto; color: #555;">Model: meta/llama-3.1-70b-instruct</span>
</div>
"""

FOOTER_HTML = """
<div class="nv-footer">
    SAP Multi-Agent Orchestrator &middot; Built with
    <a href="https://build.nvidia.com" target="_blank">NVIDIA NIM</a> &amp;
    <a href="https://gradio.app" target="_blank">Gradio</a> &middot;
    <a href="https://huggingface.co/amitlal" target="_blank">amitlal</a>
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

AGENT_EXAMPLES = {
    "ABAP_CODE": [
        "Generate a managed RAP business object for Sales Orders with draft support",
        "Write an ABAP class to call a REST API using cl_http_client and parse JSON response",
        "Refactor this SELECT...ENDSELECT loop into modern ABAP with inline declarations",
        "Create a unit test class using ABAP Unit for a purchase order validator",
    ],
    "SAP_RAG": [
        "What are the key differences between ECC and S/4HANA in the Finance module?",
        "Explain the RAP (RESTful ABAP Programming) architecture and when to use managed vs unmanaged",
        "How does SAP BTP Integration Suite connect to on-premise systems via Cloud Connector?",
        "What tables and BAPIs are used in the Procure-to-Pay process?",
    ],
    "SQL_AGENT": [
        "Show all open purchase orders over $10K with vendor name and material description",
        "Create a CDS view joining VBAK/VBAP with KNA1 for a sales order report with UI annotations",
        "Write ABAP SQL to find the top 10 vendors by total invoice amount this fiscal year",
        "Build a CDS analytical query for monthly revenue by sales organization and division",
    ],
}


def get_agent_examples_html(agent_name):
    """Return formatted HTML with sample prompts for the selected agent."""
    examples = AGENT_EXAMPLES.get(agent_name, AGENT_EXAMPLES["ABAP_CODE"])
    items = "".join(
        f'<div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:6px;'
        f'padding:10px 14px;margin:6px 0;color:#ccc;font-size:0.88em;cursor:default;">'
        f'{ex}</div>'
        for ex in examples
    )
    return f'<div style="margin-top:8px;">{items}</div>'


with gr.Blocks(
    title="SAP Multi-Agent Orchestrator | NVIDIA NIM",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#f4fce3", c100="#e6f7b3", c200="#ccef66", c300="#b3e619",
            c400="#99d900", c500="#76b900", c600="#5e9400", c700="#476f00",
            c800="#2f4a00", c900="#182500", c950="#0c1200",
        ),
        neutral_hue=gr.themes.Color(
            c50="#f5f5f5", c100="#e0e0e0", c200="#b0b0b0", c300="#888888",
            c400="#666666", c500="#444444", c600="#333333", c700="#2a2a2a",
            c800="#1a1a1a", c900="#141414", c950="#0d0d0d",
        ),
        font=["Inter", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#0d0d0d",
        body_background_fill_dark="#0d0d0d",
        block_background_fill="#141414",
        block_background_fill_dark="#141414",
        block_border_color="#2a2a2a",
        block_border_color_dark="#2a2a2a",
        block_label_text_color="#888",
        block_title_text_color="#ccc",
        input_background_fill="#1a1a1a",
        input_background_fill_dark="#1a1a1a",
        input_border_color="#333",
        input_border_color_dark="#333",
        button_primary_background_fill="#76b900",
        button_primary_background_fill_dark="#76b900",
        button_primary_text_color="#000",
        button_primary_text_color_dark="#000",
        button_secondary_background_fill="#1a1a1a",
        button_secondary_text_color="#999",
    ),
) as demo:

    gr.HTML(HEADER_HTML)
    gr.HTML(AGENTS_HTML)

    with gr.Tabs(selected="orchestrator"):
        # ---- Tab 1: Orchestrator (Auto-Route) ----
        with gr.TabItem("Orchestrator (Auto-Route)", id="orchestrator"):
            gr.HTML(STATUS_HTML)
            gr.Markdown(
                "The orchestrator analyzes your query and intelligently routes it to the best agent(s).",
            )
            chatbot = gr.Chatbot(height=480, type="messages")
            msg = gr.Textbox(
                placeholder="Ask anything about SAP — code, data, architecture, config...",
                label="Your Query",
                lines=2,
                scale=4,
            )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary", scale=2)
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)

            gr.Examples(
                examples=EXAMPLE_QUERIES,
                inputs=msg,
                label="Try these",
            )

            def respond(message, history):
                response = orchestrate(message, history)
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response})
                return "", history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            send_btn.click(respond, [msg, chatbot], [msg, chatbot])
            clear_btn.click(lambda: ([], ""), None, [chatbot, msg])

        # ---- Tab 2: Direct Agent Access ----
        with gr.TabItem("Direct Agent Access", id="direct"):
            gr.Markdown("Bypass the orchestrator — talk directly to a specialist agent.")

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
            direct_btn = gr.Button("Run Agent", variant="primary")
            direct_output = gr.Markdown(label="Agent Response")

            gr.Markdown("**Sample prompts** — copy one into the query box above:")
            examples_html = gr.HTML(value=get_agent_examples_html("ABAP_CODE"))

            agent_selector.change(
                get_agent_examples_html,
                inputs=[agent_selector],
                outputs=[examples_html],
            )

            direct_btn.click(
                run_single_agent,
                inputs=[agent_selector, direct_input],
                outputs=direct_output,
            )

        # ---- Tab 3: SAP Knowledge Explorer ----
        with gr.TabItem("SAP Knowledge Base", id="knowledge"):
            gr.Markdown("Browse the built-in SAP knowledge that powers the RAG agent.")

            with gr.Accordion("SAP Tables", open=True):
                table_data = [[k, v] for k, v in SAP_KNOWLEDGE["tables"].items()]
                gr.Dataframe(
                    value=table_data,
                    headers=["Table", "Description & Key Fields"],
                    interactive=False,
                )

            with gr.Accordion("SAP Modules", open=False):
                module_data = [[k, v] for k, v in SAP_KNOWLEDGE["modules"].items()]
                gr.Dataframe(
                    value=module_data,
                    headers=["Module", "Description"],
                    interactive=False,
                )

            with gr.Accordion("RAP Patterns", open=False):
                rap_data = [[k, v] for k, v in SAP_KNOWLEDGE["rap_patterns"].items()]
                gr.Dataframe(
                    value=rap_data,
                    headers=["Pattern", "Description"],
                    interactive=False,
                )

            with gr.Accordion("CDS Annotations", open=False):
                cds_data = [[k, v] for k, v in SAP_KNOWLEDGE["cds_annotations"].items()]
                gr.Dataframe(
                    value=cds_data,
                    headers=["Annotation", "Usage"],
                    interactive=False,
                )

        # ---- Tab 4: Architecture ----
        with gr.TabItem("Architecture", id="architecture"):
            gr.Markdown("""
### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Orchestrator Agent                          │
│         Classifies & routes queries                      │
│         Powered by NVIDIA NIM                            │
└───────┬──────────────┬──────────────┬───────────────────┘
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  ABAP Code   │ │   SAP RAG    │ │  SQL Agent   │
│    Agent     │ │    Agent     │ │              │
│              │ │              │ │              │
│  Generate    │ │  Table KB    │ │  NL → ABAP   │
│  Refactor    │ │  Module KB   │ │    SQL       │
│  Explain     │ │  RAP KB      │ │  NL → CDS    │
│  Migrate     │ │  CDS KB      │ │    views     │
└──────────────┘ └──────────────┘ └──────────────┘
        │              │              │
        ▼              ▼              ▼
┌─────────────────────────────────────────────────────────┐
│              NVIDIA NIM Inference API                     │
│           LLaMA 3.1 70B Instruct                         │
└─────────────────────────────────────────────────────────┘
```

### Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| LLM Backend | NVIDIA NIM | Free tier, fast inference, OpenAI-compatible API |
| Agent Framework | Custom lightweight | No heavy deps, HF Spaces friendly, full control |
| RAG Store | In-memory dict | Zero infra, instant startup, easy to extend |
| UI | Gradio | Native HF Spaces support, chat + direct modes |
| Routing | LLM-based + keyword fallback | Smart routing with reliable fallback |

### Extending the System

1. **Add more knowledge** — Expand `SAP_KNOWLEDGE` dict with tables, BAPIs, transactions
2. **Add new agents** — Create new `AgentType` + system prompt + route in orchestrator
3. **Add vector RAG** — Replace keyword search with NVIDIA NeMo Retriever embeddings
4. **Add ABAP execution** — Connect to SAP via ADT API for live code validation
5. **Add memory** — Store conversation context for multi-turn agent interactions
""")

    gr.HTML(FOOTER_HTML)

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(ssr_mode=False)
