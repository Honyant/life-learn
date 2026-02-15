"""FastAPI backend for the SDFT Correction Chat web UI.

Single-user server: one GPU, global state, multiple chat sessions.
Chat history persisted to SQLite via db.py.
Correction detection runs in the background (non-blocking).
Run with: uvicorn sdft_correction.server:app --host 0.0.0.0 --port 8000
"""
import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sdft_correction import db
from sdft_correction.chat import (
    _find_original_question,
    _load_env,
    _run_training_in_background,
)
from sdft_correction.config import PipelineConfig
from sdft_correction.correction_detector import detect_correction
from sdft_correction.augmenter import generate_prompt_variations
from sdft_correction.inference import LocalInference, OpenAIInference
from sdft_correction.trainer import ChatCallback


# ── Global state (training + model only, chat state lives in DB) ──
config = PipelineConfig()
llm = None
smart_llm = None
training_thread: threading.Thread | None = None
training_result: dict = {}
chat_callback: ChatCallback | None = None

# Non-blocking correction detection state
_inference_lock = None  # created in lifespan (needs running loop)
_correction_pending = False
_correction_message: str | None = None


async def _check_correction_background(chat_id: str, conversation: list):
    """Run correction detection + training setup without blocking the chat response."""
    global llm, smart_llm, training_thread, training_result, chat_callback
    global _correction_pending, _correction_message

    try:
        if training_thread is not None:
            return

        result = await asyncio.to_thread(detect_correction, conversation, smart_llm)

        if not result.is_correction:
            return

        original_question = _find_original_question(conversation)

        variations = await asyncio.to_thread(
            generate_prompt_variations,
            what_was_wrong=result.what_was_wrong,
            what_should_be=result.what_should_be,
            original_question=original_question,
            original_wrong_response=result.original_model_response,
            llm=smart_llm,
            n=config.num_prompt_variations,
        )
        all_prompts = [original_question] + variations

        # Acquire lock so we don't swap the model while another request
        # is mid-generation.  The swap itself is fast (no await inside).
        async with _inference_lock:
            if training_thread is not None:
                return  # training started while we were checking

            existing_model, existing_tokenizer = None, None
            if hasattr(llm, "extract_model_and_tokenizer"):
                existing_model, existing_tokenizer = llm.extract_model_and_tokenizer()
            tokenizer = existing_tokenizer or getattr(llm, "tokenizer", None)
            llm.unload()

            chat_cb = ChatCallback(
                tokenizer=tokenizer,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
            chat_cb.training_phase = "preparing"
            chat_callback = chat_cb
            llm = chat_cb
            if config.use_local_for_structured:
                smart_llm = llm

            training_result = {}
            training_thread = threading.Thread(
                target=_run_training_in_background,
                args=(
                    all_prompts,
                    result.what_was_wrong,
                    result.what_should_be,
                    config,
                    existing_model,
                    existing_tokenizer,
                    chat_cb,
                    training_result,
                ),
                daemon=True,
            )
            training_thread.start()

        db.reset_context(chat_id)
        _correction_message = (
            f"Correction detected! Training started with {len(all_prompts)} prompts. "
            f"You can keep chatting (responses will be slower during training)."
        )
    except Exception as e:
        print(f"[correction-bg] error: {e}")
    finally:
        _correction_pending = False


def _collect_finished_training():
    """Swap in the trained model if background training just finished."""
    global llm, smart_llm, training_thread, training_result, chat_callback
    if training_thread is not None and not training_thread.is_alive():
        training_thread = None
        if "llm" in training_result:
            llm = training_result["llm"]
            if config.use_local_for_structured:
                smart_llm = llm
        chat_callback = None
        training_result = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, smart_llm, _inference_lock
    _load_env(config)
    _inference_lock = asyncio.Lock()

    # Init database
    db.init_db()
    if not db.list_chats():
        db.create_chat("New Chat")

    trained_path = config.output_dir / "trained_model"
    if trained_path.is_dir() and (trained_path / "config.json").exists():
        print(f"[Found trained model at {trained_path}, resuming from checkpoint]")
        config.model_name = str(trained_path)
    else:
        print(f"[No trained model found, starting from {config.model_name}]")

    print("Loading model for chat...")
    llm = await asyncio.to_thread(LocalInference, config.model_name)

    if config.use_local_for_structured:
        smart_llm = llm
    else:
        smart_llm = OpenAIInference(model=config.openai_model)

    print("[Server ready]")
    yield
    print("[Shutting down]")


app = FastAPI(lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"


# ── Pydantic models ──
class ChatRequest(BaseModel):
    message: str
    chat_id: str


class ChatResponse(BaseModel):
    response: str
    is_correction: bool = False
    training_active: bool = False


class StatusResponse(BaseModel):
    training_active: bool
    progress: dict | None = None
    correction_message: str | None = None


class ResetRequest(BaseModel):
    chat_id: str


# ── Chat list endpoints ──

@app.get("/api/chats")
async def list_chats():
    return db.list_chats()


@app.post("/api/chats")
async def create_chat():
    chat = db.create_chat("New Chat")
    return {"id": chat["id"], "title": chat["title"]}


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    if not db.delete_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"status": "ok"}


@app.get("/api/chats/{chat_id}/messages")
async def get_messages(chat_id: str):
    if db.get_chat(chat_id) is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return db.get_messages(chat_id)


# ── Main chat endpoint ──

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    global llm, smart_llm, training_thread, training_result, chat_callback
    global _correction_pending

    chat_row = db.get_chat(req.chat_id)
    if chat_row is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    _collect_finished_training()

    # Persist user message
    db.add_message(req.chat_id, "user", req.message)

    # Auto-title from first user message
    all_messages = db.get_messages(req.chat_id)
    conversation = db.get_conversation(req.chat_id)
    if len(all_messages) == 1 and chat_row["title"] == "New Chat":
        title = req.message[:40]
        if len(req.message) > 40:
            title += "..."
        db.update_title(req.chat_id, title)

    # Generate response (always, immediately)
    async with _inference_lock:
        response = await asyncio.to_thread(
            llm.generate,
            conversation,
            config.max_new_tokens,
            config.temperature,
        )

        if response is None:
            # Training finished mid-request
            if training_thread is not None:
                training_thread.join()
                training_thread = None
            if "llm" in training_result:
                llm = training_result["llm"]
                if config.use_local_for_structured:
                    smart_llm = llm
            chat_callback = None
            training_result = {}
            conversation = db.get_conversation(req.chat_id)
            response = await asyncio.to_thread(
                llm.generate,
                conversation,
                config.max_new_tokens,
                config.temperature,
            )

    db.add_message(req.chat_id, "assistant", response)

    # Kick off non-blocking correction detection
    # Snapshot conversation BEFORE assistant response (that's what the detector needs)
    if (
        len(conversation) >= 3
        and training_thread is None
        and not _correction_pending
    ):
        _correction_pending = True
        asyncio.create_task(
            _check_correction_background(req.chat_id, list(conversation))
        )

    return ChatResponse(
        response=response,
        is_correction=False,
        training_active=training_thread is not None and training_thread.is_alive(),
    )


@app.get("/api/status", response_model=StatusResponse)
async def status():
    global _correction_message

    _collect_finished_training()

    active = training_thread is not None and training_thread.is_alive()
    progress = None
    if active and chat_callback is not None:
        progress = {
            "current_step": chat_callback.current_step,
            "total_steps": chat_callback.total_steps,
            "phase": chat_callback.training_phase,
        }

    # Return correction message once, then clear it
    msg = _correction_message
    _correction_message = None

    return StatusResponse(
        training_active=active or _correction_pending,
        progress=progress,
        correction_message=msg,
    )


@app.post("/api/reset")
async def reset(req: ResetRequest):
    if db.get_chat(req.chat_id) is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    db.clear_messages(req.chat_id)
    return {"status": "ok"}


@app.post("/api/save")
async def save():
    if hasattr(llm, "save"):
        await asyncio.to_thread(llm.save)
        return {"status": "saved"}
    return {"status": "no trained model to save"}


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")
