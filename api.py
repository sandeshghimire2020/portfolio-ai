import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from retriever import retrieve

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sandesh-ai")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are Sandesh Ghimire's personal AI assistant on his portfolio website sandeshghimire.net.

Answer visitor questions about Sandesh using ONLY the context provided.
- Speak naturally in third person: "Sandesh has...", "He worked at..."
- Be warm, concise, and professional
- The context may include resume-style lines like contact info (e.g. "Minneapolis, MN") — treat these as facts
- If the context contains the answer even indirectly (e.g. a city in a contact line), extract and use it
- If the context truly doesn't contain the answer, say: "Hmm, I don't have that detail — feel free to reach out to Sandesh directly at sandesh.ghimire2020@gmail.com!"
- NEVER say "based on the context" or "the documents say" — just answer naturally
- NEVER make up information not in the context"""
conversation_store: dict[str, list[dict]] = {}

DIVIDER = "-" * 60


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str


class HealthResponse(BaseModel):
    status: str
    model: str
    collection: str


def ensure_config() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in your environment or .env file.")


@app.on_event("startup")
def startup_check() -> None:
    ensure_config()
    logger.info("Startup checks passed")


def build_context_block(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{chunk['source']} | {chunk['section']}]\n{chunk['text']}" for chunk in chunks
    )


def get_or_create_history(session_id: str) -> list[dict]:
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    return conversation_store[session_id]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    logger.info(DIVIDER)
    logger.info("Incoming question: %s", req.message)
    logger.info("Session ID: %s", req.session_id)

    try:
        chunks = retrieve(req.message, top_k=5)
    except Exception as exc:
        logger.exception("Retriever failed: %s", exc)
        raise HTTPException(status_code=503, detail="Retriever unavailable") from exc

    if chunks:
        logger.info("Retrieved %s relevant chunk(s) after threshold filtering", len(chunks))
        for i, c in enumerate(chunks, 1):
            preview = c['text'][:200].replace("\n", " ")
            logger.info("Chunk %s: score=%s | %s | %s", i, c['score'], c['source'], c['section'])
            logger.info("Chunk preview: %s%s", preview, "..." if len(c['text']) > 200 else "")
    else:
        logger.warning("No chunks passed relevance threshold; continuing without retrieved context")

    if chunks:
        context_block = build_context_block(chunks)
        system_with_context = f"{SYSTEM_PROMPT}\n\nRELEVANT CONTEXT:\n{context_block}"
    else:
        system_with_context = SYSTEM_PROMPT

    logger.info("Context block sent to LLM:")
    for line in context_block.splitlines() if chunks else ["(none)"]:
        logger.info(f"    {line}")

    history = get_or_create_history(req.session_id)
    logger.info("Conversation history size: %s previous turn(s)", len(history) // 2)

    messages = (
        [{"role": "system", "content": system_with_context}]
        + history
        + [{"role": "user", "content": req.message}]
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
        )
        answer = response.choices[0].message.content or "I don't have a response right now."
    except Exception as exc:
        logger.exception("OpenAI chat completion failed: %s", exc)
        raise HTTPException(status_code=502, detail="LLM request failed") from exc

    logger.info("Assistant response: %s", answer)
    logger.info(DIVIDER)

    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": answer})
    conversation_store[req.session_id] = history[-20:]

    return ChatResponse(response=answer)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model="gpt-4o-mini",
        collection="sandesh_knowledge",
    )
