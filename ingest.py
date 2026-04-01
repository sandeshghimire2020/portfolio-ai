import os
import json
import hashlib
import pdfplumber
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
KNOWLEDGE_BASE_PATH = "./knowledge_base"
COLLECTION_NAME = "sandesh_knowledge"
EMBED_MODEL = "text-embedding-3-small"
KNOWN_JSON_KEYS = {
    "personal",
    "professional_summary",
    "technical_skills",
    "work_experience",
    "projects",
    "education",
    "nonprofit_involvement",
    "hobbies_and_interests",
    "soft_skills",
    "career_goal",
}

RESUME_SECTIONS = [
    "SUMMARY",
    "OBJECTIVE",
    "TECHNICAL SKILLS",
    "SKILLS",
    "WORK EXPERIENCE",
    "EXPERIENCE",
    "PROJECTS",
    "EDUCATION",
    "CERTIFICATIONS",
    "AWARDS",
    "OTHER INVOLVEMENTS",
    "INVOLVEMENTS",
    "VOLUNTEERING",
    "LANGUAGES",
    "INTERESTS",
    "PUBLICATIONS",
]


def embed(text: str) -> list[float]:
    response = client.embeddings.create(input=text, model=EMBED_MODEL)
    return response.data[0].embedding


def chunk_id(text: str, source: str) -> str:
    return hashlib.md5(f"{source}:{text}".encode()).hexdigest()


def load_json(filepath: str) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    filename = os.path.basename(filepath)
    chunks = []

    if "personal" in data:
        chunks.append({
            "text": f"Personal information about Sandesh:\n{json.dumps(data['personal'], indent=2)}",
            "source": filename,
            "section": "personal"
        })

    if "professional_summary" in data:
        chunks.append({
            "text": f"Professional summary:\n{data['professional_summary']}",
            "source": filename,
            "section": "professional_summary"
        })

    if "technical_skills" in data:
        chunks.append({
            "text": f"Technical skills:\n{json.dumps(data['technical_skills'], indent=2)}",
            "source": filename,
            "section": "technical_skills"
        })

    for job in data.get("work_experience", []):
        chunks.append({
            "text": f"Work experience:\n{json.dumps(job, indent=2)}",
            "source": filename,
            "section": "work_experience"
        })

    for project in data.get("projects", []):
        chunks.append({
            "text": f"Project:\n{json.dumps(project, indent=2)}",
            "source": filename,
            "section": "projects"
        })

    if "education" in data:
        chunks.append({
            "text": f"Education:\n{json.dumps(data['education'], indent=2)}",
            "source": filename,
            "section": "education"
        })

    if "nonprofit_involvement" in data:
        chunks.append({
            "text": f"Nonprofit involvement:\n{json.dumps(data['nonprofit_involvement'], indent=2)}",
            "source": filename,
            "section": "nonprofit"
        })

    extra = {}
    for key in ["hobbies_and_interests", "soft_skills", "career_goal"]:
        if key in data:
            extra[key] = data[key]
    if extra:
        chunks.append({
            "text": f"Personal interests, soft skills, and career goal:\n{json.dumps(extra, indent=2)}",
            "source": filename,
            "section": "personal_extra"
        })

    if not chunks or not any(key in data for key in KNOWN_JSON_KEYS):
        chunks.append({
            "text": json.dumps(data, indent=2),
            "source": filename,
            "section": "details"
        })

    return chunks


def is_header(line: str) -> bool:
    stripped = line.strip()
    return stripped.isupper() and 1 <= len(stripped.split()) <= 5 and stripped in RESUME_SECTIONS


def load_pdf(filepath: str) -> list[dict]:
    filename = os.path.basename(filepath)
    chunks = []

    try:
        with pdfplumber.open(filepath) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as exc:
        print(f"Warning: skipping unreadable PDF: {filename} ({exc})")
        return chunks

    lines = full_text.splitlines()

    sections: list[tuple[str, list[str]]] = []
    current_section = "header"
    current_lines: list[str] = []

    for line in lines:
        if is_header(line):
            if current_lines:
                sections.append((current_section, current_lines))
            current_section = line.strip().lower().replace(" ", "_")
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_section, current_lines))

    if len(sections) <= 1:
        paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
        for i in range(0, len(paragraphs), 3):
            group = "\n\n".join(paragraphs[i:i + 3])
            chunks.append({
                "text": group,
                "source": filename,
                "section": f"chunk_{i // 3}"
            })
        return chunks

    for section_name, section_lines in sections:
        text = "\n".join(section_lines).strip()
        if not text:
            continue
        chunks.append({
            "text": text,
            "source": filename,
            "section": section_name
        })

    return chunks


def load_text(filepath: str) -> list[dict]:
    filename = os.path.basename(filepath)
    chunks = []

    with open(filepath, encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return chunks

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    if not paragraphs:
        paragraphs = [content]

    for i in range(0, len(paragraphs), 3):
        group = "\n\n".join(paragraphs[i:i + 3])
        chunks.append({
            "text": group,
            "source": filename,
            "section": f"chunk_{i // 3}"
        })

    return chunks


def ingest_all():
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print(f"Error: knowledge_base folder not found at {KNOWLEDGE_BASE_PATH}")
        return

    files = os.listdir(KNOWLEDGE_BASE_PATH)
    if not files:
        print("Error: knowledge_base is empty. Add files and run ingestion again.")
        return

    all_chunks = []

    for filename in files:
        filepath = os.path.join(KNOWLEDGE_BASE_PATH, filename)
        ext = filename.lower().rsplit(".", 1)[-1]

        print(f"Loading file: {filename}")

        try:
            if ext == "json":
                all_chunks.extend(load_json(filepath))
            elif ext == "pdf":
                all_chunks.extend(load_pdf(filepath))
            elif ext in ("txt", "md"):
                all_chunks.extend(load_text(filepath))
            else:
                print(f"Warning: unsupported file type skipped: {filename}")
        except Exception as exc:
            print(f"Warning: failed to process {filename}: {exc}")

    print(f"\nPrepared {len(all_chunks)} chunk(s). Generating embeddings...\n")

    if not all_chunks:
        print("Error: no valid chunks were generated. Check knowledge_base files and try again.")
        return

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    for i, chunk in enumerate(all_chunks):
        embedding = embed(chunk["text"])
        doc_id = chunk_id(chunk["text"], chunk["source"])

        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "source": chunk["source"],
                "section": chunk["section"]
            }]
        )
        print(f"  [{i + 1}/{len(all_chunks)}] {chunk['source']} -> {chunk['section']}")

    print(f"\nCompleted ingestion. Stored {len(all_chunks)} chunk(s) in ChromaDB at {CHROMA_PATH}")


if __name__ == "__main__":
    ingest_all()
