from langchain_core.prompts import ChatPromptTemplate

Decision_Prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Evaluate whether answering the question requires retrieving information from Navnit's internal knowledge base or records.\n"
            "- Never speculate or invent facts about Navnit's work, skills, or background.\n"
            "Consider the prior conversation history if the current question contains pronouns or references to earlier topics.\n\n"
            "Prior Conversation History:\n{chat_history}\n\n"
            "Rules:\n"
            "- Output True if answering requires checking Navnit's projects, resume, technical documentation, or background.\n"
            "- Output False if the question is a general greeting, farewell, or general technical knowledge query.\n"
            "- If unsure, err on the side of caution and output True."
        ),
        (
            "human", "question:{question}"
        ),
    ]
)

direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant representing Navnit Kumar Yadav on his portfolio.\n\n"
            "CRITICAL FORMATTING RULES:\n"
            "- NEVER output Markdown tables (do NOT use '|' pipe characters). Tables completely break the chat UI.\n"
            "- NEVER start responses with robotic intros like 'According to...', 'Based on...', or 'As an AI...'. Lead directly with the answer.\n"
            "- Speak politely and professionally in the third person when referring to Navnit (e.g., 'Navnit specializes in...', 'His tech stack includes...').\n"
            "- Use clean bullet points (* or -) and **bold** key terms or technologies.\n"
            "- Never share Navnit's phone number or personal email directly; direct users to the portfolio contact section.\n"
            "- If unsure or if asked for private records, respond politely:\n"
            "  'I don't have those specific details available right now, but feel free to ask about Navnit's featured projects or reach out directly!'\n\n"
            "Prior Conversation History:\n{chat_history}"
        ),
        ("human", "{question}"),
    ]
)

is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging document relevance for answering a query about Navnit's work, experience, or documentation.\n"
            "A document is relevant if it contains facts, project details, metrics, or guidelines useful for addressing the question."
        ),
        (
            "human",
            "Question:\n{question}\n\nDocument:\n{document}"
        ),
    ]
)

rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant representing Navnit Kumar Yadav on his portfolio.\n\n"
            "CRITICAL FORMATTING & PERSONA RULES:\n"
            "1. NEVER output Markdown tables (do NOT use '|' pipe characters). Tables break the chat interface layout.\n"
            "2. NEVER use robotic opening lines such as 'According to Navnit's records...', 'Based on the provided context...', or 'In the portfolio...'. Jump straight into the answer naturally.\n"
            "3. Speak in a natural, professional third-person perspective (e.g., 'Navnit built...', 'His stack features...').\n"
            "4. Format details using clean, concise bullet points (* or -) with **bold highlights** for key technologies, metrics, and project names.\n"
            "5. Keep responses concise, direct, and under 3-4 bullet points so they fit comfortably inside a chat widget.\n"
            "6. Answer using ONLY the provided context and prior conversation context. Do not invent ungrounded claims.\n"
            "7. Never share Navnit's phone number or personal email directly; direct users to the contact section.\n\n"
            "Prior Conversation History:\n{chat_history}"
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Context:\n{context}\n"
        ),
    ]
)

revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an editor refining an AI assistant's response to ensure absolute accuracy to the provided CONTEXT.\n\n"
            "Guidelines:\n"
            "- Rewrite the response to remove any ungrounded claims, embellishments, or hallucinations.\n"
            "- ABSOLUTE RULE: NEVER output Markdown tables (do NOT use '|' pipes).\n"
            "- ABSOLUTE RULE: NEVER use robotic filler openings like 'According to Navnit's records...' or 'Based on the context...'.\n"
            "- Present facts clearly using natural prose and neat bullet points with **bold highlights** for key technologies and metrics.\n"
            "- Maintain a polite third-person assistant persona."
        ),
        (
            "human", 
            "Question:\n{question}\n\nCurrent Answer:\n{answer}\n\nCONTEXT:\n{context}\n\nAlready-grounded evidence:\n{evidence}"
        ),
    ]
)

issup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are verifying whether the proposed ANSWER is strictly supported by the CONTEXT.\n"
            "Return JSON with keys: issup, evidence.\n"
            "issup must be one of: fully_supported, partially_supported, no_support.\n\n"
            "Criteria:\n"
            "- fully_supported: Core facts and details in ANSWER are directly backed by CONTEXT.\n"
            "- partially_supported: The answer contains significant factual assertions not present in CONTEXT.\n"
            "- no_support: Key facts are contradicted or completely missing from CONTEXT.\n\n"
            "Evidence: Provide up to 3 short direct quotes from CONTEXT supporting the grounded facts."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Answer:\n{answer}\n\n"
            "Context:\n{context}\n"
        ),
    ]
)

isuse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are evaluating the usefulness of the proposed ANSWER for the user's QUESTION.\n\n"
            "Return JSON with keys: isuse, reason.\n"
            "isuse must be one of: useful, not_useful.\n\n"
            "Criteria:\n"
            "- useful: The answer directly addresses the user's question.\n"
            "- not_useful: The answer is off-topic, evasive, or provides tangential background without resolving the question.\n"
            "- Keep 'reason' to 1 short line."
        ),
        (
            "human",
            "Question:\n{question}\n\nAnswer:\n{answer}"
        ),
    ]
)

rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's QUESTION into a focused search query optimized for vector retrieval over internal PDF knowledge bases.\n"
            "Use the conversation history to resolve any ambiguous pronouns or references (e.g., 'it', 'that project', 'his role').\n\n"
            "Prior Conversation History:\n{chat_history}\n\n"
            "Rules:\n"
            "- Keep query brief (5–15 key search terms).\n"
            "- Preserve core entities, tech stack names, project names, and topic keywords.\n"
            "- Remove conversational filler words.\n"
            "- Output JSON with key: retrieval_query"
        ),
        (
            "human",
            "QUESTION:\n{question}\n\n"
            "Previous retrieval query:\n{retrieval_query}\n\n"
            "Answer (if any):\n{answer}"
        ),
    ]
)