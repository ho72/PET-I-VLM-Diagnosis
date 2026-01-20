"""
실행 예시:
python validation_v3.py --lora-path outputs_py/checkpoint-15020 --instruction-type json
"""

import os, glob, time, argparse, sys
from typing import List, Tuple, Optional

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="unsloth/Qwen3-VL-8B-Instruct")
    p.add_argument("--lora-path", default=None, help="LoRA 경로 입력. zeroshot을 테스트하고 싶은 경우 zeroshot 입력")
    p.add_argument("--instruction-type", default=None, help="json, markdown, zeroshot_hint 중 선택")
    p.add_argument("--image-dir", default="/workspace/eval_700/crop_padding_image")
    return p.parse_args()

# =========================
# 0) 경로/환경 설정
# =========================

args = parse_args()
BASE_MODEL_NAME = args.base_model
LORA_PATH       = args.lora_path
INSTRUCTION_TYPE = args.instruction_type
IMAGE_DIR       = args.image_dir

if LORA_PATH == None:
    print(f"[Error] --lora-path가 None입니다.\nLoRA 경로나 zeroshot을 입력해주세요.")
    sys.exit(1)
if INSTRUCTION_TYPE == None:
    print(f"[Error] --instruction-type이 None입니다.\njson, markdown, zeroshot_hint 중 하나를 입력해주세요.")
    if INSTRUCTION_TYPE not in ["json", "markdown", "zeroshot_hint"]:
        print(f"[Error] --instruction-type에 잘못된 값이 입력되었습니다.\njson, markdown, zeroshot_hint 중 선택해주세요.")

import torch
from PIL import Image

from unsloth import FastVisionModel
from transformers import AutoProcessor, GenerationConfig
from peft import PeftModel

import re, random
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from datetime import datetime
from zoneinfo import ZoneInfo


# 토크나이저 병렬 경고
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# =========================
# 1) 모델 & 프로세서 로드
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

print("[Load] Base model ...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=BASE_MODEL_NAME,
    max_seq_length=16384,
    load_in_4bit=True,
    fast_inference=False,
    gpu_memory_utilization=0.8,
)

use_lora = LORA_PATH.lower() != "zeroshot"

if use_lora:
    if not os.path.isdir(LORA_PATH):
        print(f"[Error] --lora-path로 전달된 경로가 존재하지 않습니다: {LORA_PATH}")
        sys.exit(1)
    print(f"[Load] LoRA from: {LORA_PATH}")
    model = PeftModel.from_pretrained(model, LORA_PATH)
else:
    print("[Info] Zeroshot 모드: LoRA 미적용")

model.eval()
print(f"[Ready] device={device}, IMAGE_DIR={IMAGE_DIR}")

# Processor 준비
try:
    processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
except Exception:
    processor = tokenizer

# apply_chat_template 풀백 (일부 환경 대비)
if not hasattr(processor, "apply_chat_template") and hasattr(tokenizer, "apply_chat_template"):
    processor.apply_chat_template = tokenizer.apply_chat_template   # type: ignore

# GenerationConfig 기본값
base_gen_config = GenerationConfig(
    temperature=1.0, top_k=50, max_new_tokens=1024, do_sample=True
)

print("모델 로드 완료")

# =========================
# 1-a) 프롬프트 템플릿 추가 (보고서/상담)
# =========================

diag_instruction_json = """[REPORT_DIAGNOSIS_JSON]
[SYSTEM ROLE]
당신은 동물의 안구 이미지를 분석하는 전문 수의 보조 AI입니다.
입력된 이미지를 면밀히 분석하여 [JSON 형식]으로 결과를 반환하십시오.

[지시 사항]
1. 이미지에서 관찰되는 가장 유력한 [진단명]을 도출하시오.
2. 해당 진단을 뒷받침하는 [핵심 증상]을 찾으시오.

[출력 형식 (JSON)]
반드시 아래 키(Key) 구조를 지켜야 하며, 주석이나 마크다운 코드블록(```json) 없이 순수 JSON 텍스트만 출력하시오.
{
    "diagnosis": "...",
    "symptoms": ["...", ...]
}
"""

diag_instruction_markdown = """[REPORT_DIAGNOSIS_MARKDOWN]
당신은 동물의 안구 이미지를 분석하는 전문 수의 보조 AI입니다.

[입력 설명] 분석을 위해 강아지의 눈 부위를 촬영한 사진이 제공됩니다.

[지시 사항]
1. 이미지를 분석하여 가장 가능성이 높은 '진단명'을 하나 선택하십시오.
2. 해당 진단을 내리게 된 결정적인 '증상'을 선정하십시오.
3. 각 증상에 대해 [설명]과 이미지에서 관찰되는 [시각적 특징]을 상세히 서술하십시오.
4. 만약 병변이 명확하지 않다면 '정상' 또는 '재촬영 필요'로 판단하십시오.

[출력 형식 (Markdown)]
## 진단명 : [진단명] (예: 초기 백내장, 결막염 의심 등)
## 주요 증상
### 1. [증상명]
- **설명:** [의학적 설명]
- **시각적 특징:** [사진에서 관찰되는 구체적 위치, 색상, 형태 묘사]
### 2. [증상명]
...
"""

if INSTRUCTION_TYPE == "json":
    instruction = diag_instruction_json
elif INSTRUCTION_TYPE == "markdown":
    instruction = diag_instruction_markdown
elif INSTRUCTION_TYPE == "zeroshot_hint":
    diag_sample = random.sample(["무증상", "결막염", "안검염", "안검내반증", "유루증", "백내장", "궤양성각막질환"], k=7)
    shuffle_diag = ""
    for i, llabel in enumerate(diag_sample):
        shuffle_diag += "   " + " " + llabel + "\n"
    diag_instruction_zeroshot_hint = f"""[SYSTEM ROLE]
당신은 동물의 안구 이미지를 분석하는 전문 수의 보조 AI입니다.
입력된 이미지를 면밀히 분석하여  [JSON 형식]으로 결과를 반환하십시오.

[지시 사항]
1. 이미지에서 관찰되는 가장 유력한 [진단명]을 도출하시오.
2. 해당 진단을 뒷받침하는 [핵심 증상]을 찾으시오.
3. 진단명은 다음 7가지 중에 선택하세요. ()
{shuffle_diag}
[출력 형식 (JSON)]
반드시 아래 키(Key) 구조를 지켜야 하며, 주석이나 마크다운 코드블록(```json) 없이 순수 JSON 텍스트만 출력하시오.
{{
"diagnosis": "...",
"symptoms": ["...", ...]
}}
"""
    instruction = diag_instruction_zeroshot_hint
else:
    print(f"[Error] INSTRUCTION_TYPE: {INSTRUCTION_TYPE}")

# =========================
# 2) 유틸 함수
# =========================
@torch.no_grad()
def _generate_with_image(image: Image.Image, prompt_text: str, gen_config: GenerationConfig) -> Tuple[str, float]:
    """이미지와 프롬프트 문자열로 생성 수행, (텍스트, 경과시간) 반환"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  prompt_text},
            ],
        },
    ]
    # 텍스트 프롬프트 만들기
    if hasattr(processor, "apply_chat_template"):
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif hasattr(tokenizer, "apply_chat_template"):
        text_prompt = tokenizer.apply_chat_template(messages, tokenizer=False, add_generation_prompt=True)  # type: ignore
    else:
        text_prompt = prompt_text
    
    # 입력 텐서화
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    # 생성
    start = time.time()
    outputs = model.generate(**inputs, generation_config=gen_config)
    elapsed = time.time() - start

    # 디코드 (프롬프트 제외, 새로 생성한 텍스트만)
    input_len = inputs["input_ids"].shape[1]
    gen_ids   = outputs[0, input_len:]

    tok = getattr(processor, "tokenizer", tokenizer)
    text_out = tok.decode(gen_ids, skip_special_tokens=True)
    return text_out, elapsed

@torch.no_grad()
def run_report(
    image_path: str,
    crop_image_path: str,
    report_prompt_text: str = instruction,
    temperature: float = 1.0,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    do_sample: bool = True,
) -> Tuple[str, Image.Image, str]:
    """
    단일 이미지 경로와 프롬프트로 보고서 생성.
    -> 반환 : (표시용 텍스트, PIL 이미지 객체, 원본 보고서 텍스트)
    """
    
    if not image_path or not str(image_path).strip():
        raise ValueError("이미지 경로를 입력하세요.")
    if not os.path.isfile(image_path):
        raise ValueError(f"이미지 파일을 찾을 수 없습니다 : {image_path}")
    if not report_prompt_text.strip():
        raise ValueError("보고서 프롬프트를 입력하세요.")

    image = Image.open(image_path).convert("RGB")
    gen_config = GenerationConfig(
        temperature=float(temperature),
        top_k=int(top_k),
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
    )

    report_text, elapsed = _generate_with_image(image, report_prompt_text, gen_config)

    display_text = f"{report_text}\n\n---\n⏱️ generation time: {elapsed:.2f}s"
    raw_report_text = report_text  # 상담 프롬프트에 그대로 붙일 원본
    return display_text, image, raw_report_text


# 라벨과 매핑 준비
LABELS = ["무증상", "결막염", "궤양성각막질환", "백내장", "안검내반증", "안검염", "유루증"]
NAME2ID = {name: i for i, name in enumerate(LABELS)}
ID2NAME = dict(enumerate(LABELS))

# 백업용 별칭 (영문/동의어/표기 변형)
ALIASES = {
    "결막염": ["결막염", "conjunctivitis", "Conjunctivitis"],
    "궤양성각막질환": ["궤양성각막질환", "궤양성 각막", "각막 궤양", "각막궤양", "corneal ulcer", "ulcerative-keratitis", "Ulcerative-keratitis", "Ulcerative-Keratitis"],
    "백내장": ["백내장", "cataract", "Cataract"],
    "안검염": ["안검염", "blepharitis", "Blepharitis"],
    "무증상": ["무증상", "asymptomatic", "Asymptomatic"],
    "안검내반증": ["안검내반증", "entropion", "Entropion"],
    "유루증": ["유루증", "epiphora", "Epiphora"],
}

def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower())

# 라벨 추출 규칙 : 1. 명시 패턴 2. 정규 라벨 단어 3. 별칭/영문
_PATTERNS = [
    r'(?:진단명|label)\s*[:=]\s*["\']?([^"\',\n\}\]]+)',    # 예: 진단명: 결막염 / "label": "cotaract"
    r'(?:진단명|label)\s*[\n=:\s]*["\']?([\w가-힣]+)',
    r'정답\s*[:=]\s*["\']?([^"\',\n\}\]]+)',          # 예: 정답: 각막궤양
]

# 3개 라벨을 정확하게 매칭 (한/영/한글 글자에 붙지 않도록 경계)
PAT_LABEL = re.compile(r'(?<![A-Za-z가-힣])(무증상|결막염|궤양성각막질환|백내장|안검내반증|안검염|유루증)(?![A-Za-z가-힣])')

def extract_label_from_text(text: str):
    t = text.strip()
    
    # 1. 본문 어디든 3개 라벨 찾기
    m = PAT_LABEL.search(t)
    if m:
        return NAME2ID[m.group(1)]
    
    # 2. key=value 류에서 후보 추출 후 라벨/별칭 정규화 매칭
    for p in _PATTERNS:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            # 정규 라벨 우선 매칭
            for lab in LABELS:
                if _normalize(cand) == _normalize(lab):
                    return NAME2ID[lab]
            # 별칭 매칭
            for lab, aliases in ALIASES.items():
                if any(_normalize(cand) == _normalize(a) for a in aliases):
                    return NAME2ID[lab]

    # 3. 본문에서 별칭(영문/오타 등) 포함 탐지
    low = t.lower()
    for lab, aliases in ALIASES.items():
        for a in aliases:
            if a.lower() in low:
                return NAME2ID[lab]
    
    # 실패시 None 반환
    return None

disease_map = {
    "conjunctivitis": "결막염",
    "ulcerative-keratitis": "궤양성각막질환",
    "cataract": "백내장",
    "blepharitis": "안검염",
    "Asymptomatic": "무증상",
    "Entropion": "안검내반증",
    "Epiphora": "유루증",
}

def natural_key(s):
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r'(\d+)', s)]

valid_img_list = sorted(os.listdir(IMAGE_DIR), key=natural_key)

rows = []

print(f"추론 시작 - 총 {len(valid_img_list)}개")
for img_path in tqdm(valid_img_list, desc="Inferring", unit="img"):
    try:
        # === 이미지 추론 ===
        display_text, img, raw_text = run_report(
            image_path=os.path.join(IMAGE_DIR, img_path),
            crop_image_path=os.path.join(IMAGE_DIR.replace("image", "crop_image"), img_path),
            report_prompt_text=instruction,
            temperature=1.0, top_k=50, max_new_tokens=1024, do_sample=True
        )

        # === 추론 결과에서 진단명 추출 ===
        lid = extract_label_from_text(raw_text)
        label_pred = ID2NAME.get(lid) if lid is not None else "label_not_found"

        # === 이미지 이름으로 진단명 추출 ===
        file_name = img_path.split("_")[1]
        for eng, kor in disease_map.items():
            if eng in file_name:
                label_true = file_name.replace(eng, kor)
                break
        
        # === rows에 저장 ===
        rows.append({
            "filename": img_path,
            "label_true": label_true,
            "label_pred": label_pred,
            "pred_text": raw_text,
        })
    except Exception as e:
        print(f"[Error] - {img_path} - {e}")
        rows.append({
            "filename": img_path,
            "label_true": None,
            "label_pred": None,
            "pred_text": f"[Error] {e}",
        })

print(f"✅ 완료: {len(rows)}/{len(valid_img_list)}개 처리 끝!")

def save_results_to_excel(rows, labels, out_path="result_v2.xlsx"):
    """
    rows: [{"filename":..., "label_true":..., "label_pred":..., "pred_text":...}, ...]
    labels: 라벨 순서 고정 리스트 (예: ["결막염", "궤양성각막질환", ...])
    out_path: 저장 경로
    """
    # ---------- 0) DataFrame ----------
    df = pd.DataFrame(rows)
    # 필요 시 결측 제거
    df = df.dropna(subset=["label_true", "label_pred"])

    df_valid = df[df["label_pred"].isin(labels)]
    coverage = len(df_valid)/len(df) if len(df) else 0.0

    y_true = df_valid["label_true"].tolist()
    y_pred = df_valid["label_pred"].tolist()
    # ---------- 1) Metrics ----------
    acc = accuracy_score(y_true, y_pred)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="micro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    # 요약표
    summary_df = pd.DataFrame({
        "metric": [
            "accuracy",
            "coverage",
            "precision_macro", "recall_macro", "f1_macro",
            "precision_micro", "recall_micro", "f1_micro",
            "precision_weighted", "recall_weighted", "f1_weighted",
        ],
        "value": [
            acc,
            coverage,
            p_macro, r_macro, f1_macro,
            p_micro, r_micro, f1_micro,
            p_weighted, r_weighted, f1_weighted,
        ],
    })

    # 클래스별 상세 리포트
    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_class_df = pd.DataFrame(report_dict).T
    # 순서 정렬: 클래스들만 먼저
    per_class_df = per_class_df.loc[[*labels, "accuracy", "macro avg", "weighted avg"]]
    per_class_df = per_class_df.rename_axis("class").reset_index()

    # 혼동행렬
    cm_labels = labels + ["label_not_found"]
    y_pred_all = df["label_pred"].apply(lambda x: x if x in labels else "label_not_found").tolist()
    cm = confusion_matrix(df["label_true"].tolist(), y_pred_all, labels=cm_labels)
    cm_df = pd.DataFrame(cm, index=[f"T:{c}" for c in cm_labels], columns=[f"P:{c}" for c in cm_labels])

    # ---------- 2) Excel 쓰기 ----------
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        # 시트1: 원본 예측 rows
        df.to_excel(writer, sheet_name="Predictions", index=False)

        # 시트2: 지표들
        ws = writer.sheets["Metrics"] = writer.book.add_worksheet("Metrics")

        # 포맷들
        book = writer.book
        fmt_title   = book.add_format({"bold": True, "font_size": 14})
        fmt_header  = book.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
        fmt_num3    = book.add_format({"num_format": "0.000"})
        fmt_int     = book.add_format({"num_format": "0"})
        fmt_box     = book.add_format({"border": 1})
        fmt_note    = book.add_format({"font_color": "#666666", "italic": True})

        row = 0
        # (A) Summary
        ws.write(row, 0, "Summary Metrics", fmt_title); row += 2
        summary_df_rounded = summary_df.copy()
        summary_df_rounded["value"] = summary_df_rounded["value"].round(4)
        summary_df_rounded.to_excel(writer, sheet_name="Metrics", startrow=row, startcol=0, index=False)
        # 헤더 스타일 적용
        for col, col_name in enumerate(summary_df_rounded.columns):
            ws.write(row, col, col_name, fmt_header)
        # 값 서식
        ws.set_column(0, 0, 22)  # metric
        ws.set_column(1, 1, 14, fmt_num3)  # value
        row += len(summary_df_rounded) + 3

        # (B) Per-class report
        ws.write(row, 0, "Per-class Report (precision / recall / f1-score / support)", fmt_title); row += 2
        per_class_print = per_class_df.copy()
        # 반올림
        for c in ["precision", "recall", "f1-score"]:
            if c in per_class_print.columns:
                per_class_print[c] = per_class_print[c].astype(float).round(4)
        per_class_print["support"] = per_class_print["support"].astype(int)
        per_class_print.to_excel(writer, sheet_name="Metrics", startrow=row, startcol=0, index=False)

        # 헤더 스타일 + 수치 서식
        for col, col_name in enumerate(per_class_print.columns):
            ws.write(row, col, col_name, fmt_header)
        # 자동 열 너비
        def _autofit(ws, df, start_row, start_col):
            for j, col in enumerate(df.columns):
                series = df[col].astype(str)
                max_len = max([len(str(col))] + series.map(len).tolist())
                ws.set_column(start_col + j, start_col + j, min(max_len + 2, 40))
        _autofit(ws, per_class_print, row, 0)
        row += len(per_class_print) + 3

        # (C) Confusion Matrix (heatmap)
        ws.write(row, 0, "Confusion Matrix", fmt_title); row += 2

        # 테두리 포함 포맷
        fmt_header_b = book.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
        fmt_int_b    = book.add_format({"num_format": "0", "border": 1})

        # 헤더
        ws.write(row, 0, "", fmt_header_b)
        for j, c in enumerate(cm_df.columns, start=1):
            ws.write(row, j, c, fmt_header_b)

        # 바디 (값을 테두리포맷으로 직접 씀)
        for i, (idx, r) in enumerate(cm_df.iterrows(), start=1):
            ws.write(row + i, 0, idx, fmt_header_b)     # 행 헤더
            for j, v in enumerate(r.values, start=1):
                ws.write(row + i, j, int(v), fmt_int_b) # 데이터(숫자+테두리)

        # 히트맵 (데이터 영역만)
        h, w = cm_df.shape
        first_data_row = row + 1
        first_data_col = 1
        last_data_row  = row + h
        last_data_col  = w
        ws.conditional_format(first_data_row, first_data_col, last_data_row, last_data_col, {
            "type": "3_color_scale",
            "min_color": "#FFFFFF",
            "mid_color": "#FFD966",
            "max_color": "#9DC3E6",
        })

        # 열 너비
        ws.set_column(0, 0, 11)
        for j in range(1, w + 1):
            ws.set_column(j, j, 13)

        # (D) 주석
        row = last_data_row + 2
        ws.write(row, 0, "Note: zero_division=0 → 예측이 전혀 없던 클래스의 precision/recall은 0으로 처리됩니다.", fmt_note)

        # 시트1(원본): 필터, freeze
        ws_pred = writer.sheets["Predictions"]
        # 자동열너비
        def _autofit_df_sheet(ws, dframe, start_col=0):
            for j, col in enumerate(dframe.columns):
                series = dframe[col].astype(str)
                max_len = max([len(str(col))] + series.map(len).tolist())
                ws.set_column(start_col + j, start_col + j, min(max_len + 2, 60))
        _autofit_df_sheet(ws_pred, df)
        # 필터/고정
        ws_pred.autofilter(0, 0, len(df), len(df.columns)-1)
        ws_pred.freeze_panes(1, 0)

    print(f"✅ Excel saved to: {out_path}")

def safe_slug(s: str, keep: int = 80) -> str:
    """파일명에 안전하도록 치환 + 길이 제한"""
    s = os.path.basename(s)           # 경로 제거
    s = re.sub(r"[\\/:*?\"<>|]", "-", s)  # 금지문자 치환
    s = re.sub(r"\s+", "_", s)        # 공백 → _
    return s[:keep]                   # 너무 길면 자르기

stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S")

lora_tag = "zeroshot" if LORA_PATH.lower() == "zeroshot" else safe_slug(LORA_PATH)
save_name = f"val_{lora_tag}_{stamp}.xlsx"

save_results_to_excel(rows, LABELS, out_path=save_name)