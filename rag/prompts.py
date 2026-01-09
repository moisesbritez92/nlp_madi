"""
Prompts
=======
System prompts and formatting functions for RAG pipeline.
"""

from typing import Dict, List, Tuple


# ============================================
# System Prompt for RAG
# ============================================
SYSTEM_PROMPT = """You are a helpful assistant. You will receive a set of relevant documents and a user question.

**IMPORTANT RULES:**
1. Answer the question using ONLY the information in the provided documents.
2. If the documents do not contain enough information to answer, say so explicitly.
3. Cite supporting documents inline using their IDs, e.g. [doc1], [doc2].
4. Do NOT invent facts or use external knowledge.
5. Be concise, clear, and use the same language as the user.
6. If multiple documents support your answer, cite all relevant ones.

**RELEVANT DOCUMENTS:**
{documents}
"""

SYSTEM_PROMPT_ES = """Eres un asistente útil. Recibirás un conjunto de documentos relevantes y una pregunta del usuario.

**REGLAS IMPORTANTES:**
1. Responde la pregunta usando SOLO la información de los documentos proporcionados.
2. Si los documentos no contienen suficiente información para responder, indícalo explícitamente.
3. Cita los documentos de apoyo usando sus IDs, ej. [doc1], [doc2].
4. NO inventes datos ni uses conocimiento externo.
5. Sé conciso, claro y usa el mismo idioma que el usuario.
6. Si múltiples documentos apoyan tu respuesta, cita todos los relevantes.

**DOCUMENTOS RELEVANTES:**
{documents}
"""


def format_documents_for_prompt(
    documents: List[Tuple[str, Dict, float]]
) -> str:
    """
    Format retrieved documents for inclusion in the LLM prompt.
    
    Args:
        documents: List of (doc_id, doc_content, score) tuples
                  where doc_content is {"title": ..., "text": ...}
        
    Returns:
        Formatted string for prompt insertion
    """
    if not documents:
        return "(No documents retrieved)"
    
    formatted_parts = []
    
    for doc_id, doc, score in documents:
        title = doc.get("title", "").strip()
        text = doc.get("text", "").strip()
        
        if title:
            formatted_parts.append(f"[{doc_id}] Title: {title}\nText: {text}")
        else:
            formatted_parts.append(f"[{doc_id}] {text}")
    
    return "\n\n".join(formatted_parts)


def build_rag_messages(
    query: str,
    documents: List[Tuple[str, Dict, float]],
    language: str = "auto"
) -> List[Dict[str, str]]:
    """
    Build the message list for RAG LLM call.
    
    Args:
        query: User query
        documents: Retrieved documents with scores
        language: "en", "es", or "auto" (detect from query)
        
    Returns:
        List of messages for LLM API call
    """
    # Format documents
    docs_text = format_documents_for_prompt(documents)
    
    # Select language for system prompt
    if language == "auto":
        # Simple heuristic: check for Spanish characters/words
        spanish_indicators = ["¿", "¡", "ñ", " es ", " el ", " la ", " que ", " de "]
        query_lower = query.lower()
        is_spanish = any(ind in query_lower for ind in spanish_indicators)
        prompt_template = SYSTEM_PROMPT_ES if is_spanish else SYSTEM_PROMPT
    elif language == "es":
        prompt_template = SYSTEM_PROMPT_ES
    else:
        prompt_template = SYSTEM_PROMPT
    
    # Build system message
    system_content = prompt_template.format(documents=docs_text)
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]


def format_prompt_for_display(messages: List[Dict[str, str]]) -> str:
    """
    Format messages for display in UI (debug mode).
    
    Args:
        messages: List of chat messages
        
    Returns:
        Formatted string for display
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        parts.append(f"--- {role} ---\n{content}")
    
    return "\n\n".join(parts)


# ============================================
# Fallback Response
# ============================================
def get_fallback_response(
    documents: List[Tuple[str, Dict, float]],
    error_msg: str
) -> str:
    """
    Generate a fallback response when LLM fails.
    
    Args:
        documents: Retrieved documents
        error_msg: Error message from LLM call
        
    Returns:
        Fallback response string
    """
    response = f"⚠️ **Could not generate LLM response**: {error_msg}\n\n"
    response += "**However, here are the retrieved documents:**\n\n"
    
    for doc_id, doc, score in documents:
        title = doc.get("title", "")
        text = doc.get("text", "")
        preview = text[:200] + "..." if len(text) > 200 else text
        
        response += f"**[{doc_id}]** (score: {score:.4f})\n"
        if title:
            response += f"*{title}*\n"
        response += f"{preview}\n\n"
    
    return response
