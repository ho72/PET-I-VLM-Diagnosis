# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import base64
import httpx
import hashlib

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="PET-I Backend")

CASES: dict[str, dict] = {}
# ======== 환경 변수 =========
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync" if RUNPOD_ENDPOINT_ID else ""

# ======== CORS 설정 =========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 필요하면 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======== 업로드 설정 =========
ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
MAX_FILE_MB = 5
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

# ======== 요청 모델 =========
class ChatRequest(BaseModel):
    case_id: str
    message: str
    answer_mode: str | None = "brief"   # 추후 "detail" 추가 가능


# ======== 진단 엔드포인트 =========
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    case_id: str | None = Form(None),
):
    """
    Flutter에서:
      multipart/form-data:
        - image: 파일
        - case_id: (선택, 없으면 서버에서 기본값 사용)

    Runpod handler.py (mode='diag')로 요청:
      {
        "input": {
          "mode": "diag",
          "images": ["<base64>"],
          "case_id": "<선택>"
        }
      }
    """
    # 1. 파일 타입 검사
    if image.content_type not in ALLOWED_TYPES:
        return JSONResponse(
            status_code=415,
            content={
                "status": "error",
                "code": "BAD_INPUT",
                "message": f"Only {', '.join(ALLOWED_TYPES)} supported",
            },
        )

    # 2. 파일 크기 검사
    content = await image.read()
    if len(content) > MAX_FILE_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "status": "error",
                "code": "FILE_TOO_LARGE",
                "message": f"File exceeds {MAX_FILE_MB}MB limit",
            },
        )

    if not RUNPOD_URL or not RUNPOD_API_KEY:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "code": "RUNPOD_CONFIG_MISSING",
                "message": "Runpod endpoint or API key is not configured.",
            },
        )

    # 3. Base64 인코딩
    b64_img = base64.b64encode(content).decode("utf-8")

    # 4. Runpod payload (새 handler.py 기준)
    payload = {
        "input": {
            "mode": "diag",
            "images": [b64_img],
        }
    }
    if case_id:
        payload["input"]["case_id"] = case_id

    # 5. Runpod 호출
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                RUNPOD_URL,
                json=payload,
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            )
    except httpx.ReadTimeout:
        return JSONResponse(
            status_code=504,
            content={
                "status": "error",
                "code": "RUNPOD_TIMEOUT",
                "message": "Runpod request timed out.",
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={
                "status": "error",
                "code": "RUNPOD_CALL_FAILED",
                "message": f"{type(e).__name__}: {e}",
            },
        )

    # 6. Runpod 응답 파싱
    raw = response.json()
    # runsync 기본 형태: {"id": ..., "status": "...", "output": {...}}
    core = raw.get("output") or raw.get("result") or raw

    if isinstance(core, dict):
        report_md = core.get("output", "")
        diagnosis = core.get("diagnosis")
        case_id_resp = core.get("case_id")
        mode = core.get("mode", "diag")
    else:
        # 혹시 dict가 아니더라도 최소한 report 텍스트는 뽑아줌
        report_md = str(core)
        diagnosis = None
        case_id_resp = case_id
        mode = "diag"

    # 이미지 해시 (추적 용도)
    image_hash = hashlib.sha256(content).hexdigest()[:16]

    case_id_key = case_id_resp or case_id or image_hash

    if diagnosis is not None or report_md:
        CASES[case_id_key] = {
            "diagnosis": diagnosis,
            "report_markdown": report_md,
        }

    return {
        "status": "ok" if response.status_code == 200 else "runpod_error",
        "upstream_status": response.status_code,
        "data": {
            "report_markdown": report_md,
            "diagnosis": diagnosis,
            "case_id": case_id_key,
            "mode": mode,
            "image_hash": image_hash,
        },
    }


# ======== 챗봇 엔드포인트 =========
@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Flutter에서:
      POST /chat
      JSON body:
        {
          "case_id": "<diag 결과로 받은 case_id>",
          "message": "질문 내용",
          "answer_mode": "brief"  // 선택, 기본은 "brief"
        }

    Runpod handler.py (mode='chat')로 요청:
      {
        "input": {
          "mode": "chat",
          "case_id": "<case_id>",
          "question": "<질문>",
          "answer_mode": "brief"
        }
      }
    """
    if not RUNPOD_URL or not RUNPOD_API_KEY:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "code": "RUNPOD_CONFIG_MISSING",
                "message": "Runpod endpoint or API key is not configured.",
            },
        )
    
    case = CASES.get(req.case_id)
    if not case:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "code": "CASE_NOT_FOUND",
                "message": f"Case ID '{req.case_id}' not found.",
            },
        )
    
    diagnosis = case.get("diagnosis")
    report_md = case.get("report_markdown")

    payload = {
        "input": {
            "mode": "chat",
            "case_id": req.case_id,
            "question": req.message,
            "answer_mode": req.answer_mode or "brief",
            "diagnosis": diagnosis,
            "report_markdown": report_md,
        }
    }

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                RUNPOD_URL,
                json=payload,
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            )
    except httpx.ReadTimeout:
        return JSONResponse(
            status_code=504,
            content={
                "status": "error",
                "code": "RUNPOD_TIMEOUT",
                "message": "Runpod request timed out.",
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={
                "status": "error",
                "code": "RUNPOD_CALL_FAILED",
                "message": f"{type(e).__name__}: {e}",
            },
        )

    raw = response.json()
    core = raw.get("output") or raw.get("result") or raw

    if isinstance(core, dict):
        answer = (
            core.get("output")
            or core.get("answer")
            or core.get("data")
            or core
        )
        case_id_resp = core.get("case_id", req.case_id)
        mode = core.get("mode", "chat")
        answer_mode = core.get("answer_mode", req.answer_mode or "brief")
    else:
        answer = core
        case_id_resp = req.case_id
        mode = "chat"
        answer_mode = req.answer_mode or "brief"

    # answer가 dict면 문자열로 변환
    if isinstance(answer, dict):
        answer = answer.get("output") or str(answer)

    return {
        "status": "ok" if response.status_code == 200 else "runpod_error",
        "upstream_status": response.status_code,
        "data": {
            "answer": answer,
            "case_id": case_id_resp,
            "mode": mode,
            "answer_mode": answer_mode,
        },
    }


@app.get("/")
def root():
    return {"msg": "FASTAPI OK"}
