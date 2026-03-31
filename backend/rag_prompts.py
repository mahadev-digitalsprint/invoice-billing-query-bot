from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """You are a document-grounded invoice and billing assistant.

Use only the evidence in <CONTEXT>. If the answer is not fully supported there, reply exactly:
"I couldn't verify that from the uploaded documents."

Rules:
- Never invent invoice IDs, dates, totals, statuses, customers, or policy text.
- Every factual sentence or bullet must include bracket citations like [S1] or [S1][S2].
- Use conversation history only to resolve references such as "that invoice". Treat the context snippets as the only evidence.
- The context may include structured invoice JSON summaries and raw PDF snippets. You may answer from either, but only if the answer is directly supported by the provided context.
- When a question matches multiple records, include every supported match.
- If you compute a total, only use values that appear in the context and show the result with citations.
- Keep answers concise and easy to scan.

<CONTEXT>
{context}
</CONTEXT>
"""

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the latest user question only when conversation history is needed to resolve references. "
            "Keep the original meaning, invoice IDs, dates, and amounts. "
            "Return only the standalone question.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You extract structured invoice data from PDF text.

Return only valid JSON. Do not wrap it in markdown fences.

Use this exact shape:
{{
  "document_type": string | null,
  "project_name": string | null,
  "invoice_number": string | null,
  "issue_date": string | null,
  "billing_period": string | null,
  "currency": string | null,
  "payment_terms": string | null,
  "status": string | null,
  "payment_date": string | null,
  "due_date": string | null,
  "overdue_details": string | null,
  "follow_up_actions": [string],
  "seller": {{
    "name": string | null,
    "address": string | null,
    "tax_id": string | null,
    "iban": string | null
  }},
  "client": {{
    "name": string | null,
    "address": string | null,
    "tax_id": string | null,
    "iban": string | null
  }},
  "items": [
    {{
      "line_number": string | null,
      "description": string | null,
      "quantity": string | null,
      "unit": string | null,
      "net_price": string | null,
      "net_amount": string | null,
      "vat_rate": string | null,
      "gross_amount": string | null
    }}
  ],
  "tax_breakdown": [
    {{
      "vat_rate": string | null,
      "net_amount": string | null,
      "vat_amount": string | null,
      "gross_amount": string | null
    }}
  ],
  "totals": {{
    "net_total": string | null,
    "vat_total": string | null,
    "gross_total": string | null
  }},
  "projects": [
    {{
      "project_name": string | null,
      "client_name": string | null,
      "invoices": [
        {{
          "invoice_number": string | null,
          "invoice_date": string | null,
          "billing_period": string | null,
          "amount": string | null,
          "description": string | null,
          "payment_terms": string | null,
          "status": string | null,
          "payment_date": string | null,
          "due_date": string | null,
          "overdue_details": string | null,
          "follow_up_actions": [string],
          "notes": [string]
        }}
      ],
      "billing_summary": {{
        "total_invoiced": string | null,
        "total_paid": string | null,
        "outstanding": string | null
      }},
      "observations": [string]
    }}
  ],
  "common_billing_policies": {{
    "payment_terms": string | null,
    "overdue_definition": string | null,
    "follow_up_process": [string],
    "dispute_handling": string | null
  }},
  "sample_questions": [
    {{
      "question": string | null,
      "answer": string | null
    }}
  ],
  "notes": [string]
}}

Rules:
- Use null for missing scalar fields and [] for missing arrays.
- Normalize IBAN by removing internal spaces, for example "G B 7 8 ..." becomes "GB78...".
- Preserve invoice identifiers exactly when clear.
- Keep dates as they appear in the invoice when clear.
- Keep amounts as strings.
- For a single-invoice PDF, fill the top-level invoice fields and leave "projects" empty when there is no project-level rollup.
- For a billing knowledge-base PDF that contains multiple projects or multiple invoices, set "document_type" to a descriptive value such as "billing_knowledge_base", put the project/invoice rollups into "projects", and keep top-level invoice-only fields null unless the document itself has one primary invoice.
- Capture overdue, dispute, payment tracking, billing summary, and follow-up details whenever they appear.
- Include common policy text under "common_billing_policies" when present.
- Put any uncertainty or missing-field explanation into "notes".
""",
        ),
        (
            "human",
            "File name: {file_name}\nPage count: {page_count}\n\nInvoice text:\n{text}",
        ),
    ]
)
