# ===== console setup =====
from rich import box
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich.traceback import install as rich_traceback_install
try:
    from transformers.integrations import RichProgressCallback  # HF 진행바를 rich 스타일로
    _HAS_RICH_PROGRESS = True
except Exception:
    RichProgressCallback = None
    _HAS_RICH_PROGRESS = False

import os
os.environ.setdefault("TERM", "xterm-256color")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
_RICH_TQDM = None
try:
    from tqdm.rich import tqdm as _rich_tqdm
    _RICH_TQDM = _rich_tqdm
except Exception:
    try:
        from tqdm.contrib.rich import tqdm as _rich_tqdm
        _RICH_TQDM = _rich_tqdm
    except Exception:
        _RICH_TQDM = None

if _RICH_TQDM is not None:
    import tqdm.auto as _tqa
    _tqa.tqdm = _RICH_TQDM
else:
    pass

# 콘솔 및 traceback 활성화
console = Console(highlight=False)
rich_traceback_install(show_locals=False)   # 에러 스택

# 섹션 배너
def banner(title: str, emoji: str = "🚀", style: str = "bold cyan"):
    console.rule(f"{emoji}  {title}", style=style)

# Key-Value 테이블
def kv_table(title: str, pairs: dict, style="bold white on blue"):
    t = Table.grid(padding=1)
    t.add_column(justify="right", style="bold cyan")
    t.add_column()
    for k, v in pairs.items():
        t.add_row(str(k), str(v))
    console.print(Panel(t, title=title, title_align="left", style=style, border_style="cyan"))

def ok(msg: str):     console.print(f"✅ [bold green]{msg}[/]")
def warn(msg: str):   console.print(f"⚠️  [bold yellow]{msg}[/]")
def info(msg: str):   console.print(f"📝 {msg}")
def fail(msg: str):   console.print(f"❌ [bold red]{msg}[/]")

# 간단 스피너 컨텍스트
from contextlib import contextmanager
@contextmanager
def step(title: str, spinner="dots", color="cyan"):
    with console.status(f"[{color}]{title}[/] …", spinner=spinner, spinner_style=color):
        yield
    ok(title + " 완료")

banner("Preparing Environment", "🧰")

import glob, json, copy, unicodedata, jsonlines
from datasets import load_dataset, Features, Value, Sequence
from PIL import Image

import unsloth
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TrainingArguments, AutoProcessor, GenerationConfig
from transformers import Trainer
from peft import PeftModel
import torch

dataset_path = "./train_original"
wandb_project = "sft-finetuning-project_v2"
wandb_run = "training_v003_qwen3-8B(single_img,all_data)"

print("===== load unsloth =====")
banner("Load Unsloth", "⚙️")
kv_table("Run Config", {
    "dataset_path": dataset_path,
    "wandb_project": wandb_project,
    "run_name": wandb_run,
})

max_seq_length = 16384
lora_rank = 16
MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct"

with step("모델/토크나이저 로드"):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = max_seq_length,
        load_in_4bit = True,            # False for LoRA 16bit
        fast_inference = False,         # Enable vLLM fast inference
        gpu_memory_utilization = 0.8,   # Reduce if out of memory
        #vision_processor_image_size = 512,
    )

with step("프로세서 로드"):
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code = True,
    )

with step("LoRA 어댑터 구성"):
    # 로드한 모델 객체를 받아 그 위에 LoRA 레이어(어댑터)를 추가한 새로운 모델 객체를 반환
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers      = True,
        finetune_language_layers    = True,
        finetune_attention_modules  = True,
        finetune_mlp_modules        = True,

        r = lora_rank,
        lora_alpha = lora_rank,
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
        use_gradient_checkpointing = "unsloth"
    )

banner("Start Preparing Dataset", "📦")

info("학습 데이터셋 확인")
jsonl_path = "/workspace/train_all.jsonl"
jsonl_data_list = []
with jsonlines.open(jsonl_path) as f:
    for line in f.iter():
        jsonl_data_list.append(line)

info(f"총 파일 수: [bold]{len(jsonl_data_list)}[/]")
tbl = Table(title="Dataset Files (sample)", box=box.SIMPLE, show_header=True, header_style="bold magenta")
tbl.add_column("#", justify="right", width=3)
tbl.add_column("path", overflow="fold")
for i, p in enumerate(jsonl_data_list[:5], 1):
    tbl.add_row(str(i), str(p))
console.print(tbl)
console.print(Rule(style="dim"))

banner("데이터 로드 및 전처리 시작", "🧪")

features = Features({
    "text_name": Value("string"),
    "normalized": {
        "diagnosis": Value("string"),
        "symptoms": Sequence(Value("string")),
    },
    "items": [{
        "template_id": Value("string"),
        "instruction": Value("string"),
        "answer": Value("string"),
        "token": Value("string"),
    }],
})

with step("load_dataset 호출"):
    dataset = load_dataset(
        "json",
        data_files = jsonl_path,
    )

ok("데이터 로드 완료!")

diag_instruction = """[diagnosis_report]
당신은 반려동물의 질환 진단을 위한 수의사입니다.
사진을 보고 반려동물의 안구 질환을 진단하여 진단 리포트를 작성하세요.
‘# 진단명’, ‘## 증상’ 헤더를 포함하세요.
형식은 마크다운 형식으로 작성해주세요.
"""
chat_instruction = """[chatbot_consultation]
당신은 반려동물의 질환 진단을 상담해주는 AI 수의사입니다.
다음의 진단 결과를 바탕으로 친절하게 상담해주세요.
"""

disease_map = {
    "결막염": "conjunctivitis",
    "궤양성각막질환": "ulcerative-keratitis",
    "백내장": "cataract",
    "안검염": "blepharitis",
    "무증상": "Asymptomatic",
    "안검내반증": "Entropion",
    "유루증": "Epiphora",
}

def kor2eng(name):
    normalized_name = unicodedata.normalize("NFC", name)

    for kor, eng in disease_map.items():
        if unicodedata.normalize("NFC", kor) in normalized_name:
            normalized_name = normalized_name.replace(kor, eng)

    return normalized_name

def img_format(json_file):
    img_file = os.path.join(os.path.join(dataset_path, "image"), json_file)

    if os.path.exists(img_file.replace('.json', '.jpg')):
        return img_file.replace('.json', '.jpg')
    elif os.path.exists(img_file.replace('.json', '.jpeg')):
        return img_file.replace('.json', '.jpeg')
    elif os.path.exists(img_file.replace('.json', '.png')):
        return img_file.replace('.json', '.png')
    else:
        warn(f"이미지 파일 없음: {json_file}")
        return None

def flatten_items(batch):
    new_data = {
        'json_file_name': [],
        'img_file_path': [],
        'diagnosis': [],
        'symptoms': [],
        'instruction': [],
        'answer': [],
        'token': []
    }

    for i in range(len(batch['text_name'])):
        text_name = kor2eng(batch["text_name"][i])

        if text_name.endswith('.txt'):
            json_file_name = text_name.replace('.txt', '.json')
        elif text_name.endswith('.text'):
            json_file_name = text_name.replace('.text', '.json')
        elif text_name.endswith('.json'):
            json_file_name = text_name

        img_file_path = img_format(json_file_name)
        diagnosis = batch['normalized'][i]['diagnosis']
        symptoms = batch['normalized'][i]['symptoms']

        for item in batch['items'][i]:
            new_data['json_file_name'].append(json_file_name)
            new_data['img_file_path'].append(img_file_path)
            new_data['diagnosis'].append(diagnosis)
            new_data['symptoms'].append(symptoms)

            new_data['token'].append(item['token'])
            new_data['instruction'].append(item['instruction'])
            new_data['answer'].append(item['answer'])
    
    return new_data

with step("flatten_items 매핑"):
    flattened_dataset = dataset['train'].map(
        flatten_items,
        batched = True,
        remove_columns = dataset['train'].column_names
    )

def create_training(data):

    instruction = data.get("instruction", "")
    answer_text = data.get("answer", "")
    img_path = data.get("img_file_path", "").replace("/image/", "/crop_padding_image/")

    instruction = instruction if isinstance(instruction, str) else str(instruction)
    answer_text = answer_text if isinstance(answer_text, str) else str(answer_text)
    img_path = img_path if isinstance(img_path, str) else str(img_path)

    for t in ("<image>", "<|image|>", "<|vision_start|>", "<|vision_end|>"):
        instruction = instruction.replace(t, "")
        answer_text = answer_text.replace(t, "")

    if img_path and not os.path.isabs(img_path):
        img_path = os.path.abspath(img_path)

    user_content = []
    if img_path and os.path.exists(img_path):
        user_content.append({"type": "image", "image_url": img_path})
    user_content.append({"type": "text", "text": instruction})

    conversation =[
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
    ]

    return {"messages": conversation}

with step("create_training 매핑"):
    processed_dataset = flattened_dataset.map(
        create_training,
        num_proc = 1,
        remove_columns = flattened_dataset.column_names
    )

banner("데이터 전처리 완료", "✅")

# ===== Train =====

# --- Collator ---

IMAGE_TOKENS = ("<image>", "<|image|>")
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

def _sanitize_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    for t in (IM_START, IM_END, "<|vision_start|>", "<|vision_end|>") + IMAGE_TOKENS:
        s = s.replace(t, "")
    return s

def _clean_messages(messages):
    cleaned = []
    for msg in messages:
        new_content = []
        
        for part in msg.get("content", []):
            part = {k: v for k, v in part.items() if v is not None}
            t = part.get("type")

            if t == "image":
                url = part.get("image_url")
                if not url:
                    continue
                if not os.path.isabs(url):
                    url = os.path.abspath(url)
                if not os.path.exists(url):
                    continue
                new_content.append({"type": "image", "image_url": url})

            elif t == "text":
                txt = _sanitize_text(part.get("text"))
                if txt == "":
                    pass
                else:
                    new_content.append({"type": "text", "text": txt})

        if new_content:
            cleaned.append({"role": msg.get("role", "user"), "content": new_content})

    return cleaned

class FixMaskCollator:
    
    def __init__(self, model, processor, require_user_image=True, require_asst_text=True):
        self.base = UnslothVisionDataCollator(model, processor)
        self.processor = processor
        self.tok = processor.tokenizer
        self.require_user_image = require_user_image
        self.require_asst_text = require_asst_text

        image_token_ids = []
        for t in IMAGE_TOKENS:
            tid = self.tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid != self.tok.unk_token_id:
                image_token_ids.append(tid)
        self.image_token_ids = set(image_token_ids)

        self.im_start_id    = self.tok.convert_tokens_to_ids(IM_START)
        self.im_end_id      = self.tok.convert_tokens_to_ids(IM_END)
        self.assistant_ids  = self.tok.encode("assistant", add_special_tokens=False)
        
        # 공백/개행 토큰 집합
        ws = set()
        for ch in [" ", "\n", "\t", "\r"]:
            ids = self.tok.encode(ch, add_special_tokens=False)
            for i in ids:
                ws.add(i)
        self.ws_ids = ws

    @staticmethod
    def _has_text(parts):
        return any(p.get("type")=="text" and p.get("text") not in ("", None) for p in parts)
    
    @staticmethod
    def _count_images(parts):
        return sum(1 for p in parts if p.get("type")=="image")
    
    def _assistant_mask_from_ids_tokenwise(self, ids_tensor: torch.Tensor) -> torch.Tensor:
        ids = ids_tensor.tolist()
        T = len(ids)
        mask = torch.zeros(T, dtype=torch.bool, device=ids_tensor.device)

        i = 0
        while i < T:
            # im_start 찾기
            while i < T and ids[i] != self.im_start_id:
                i += 1
            if i >= T: break
            i += 1

            # 공백/개행 스킵
            while i < T and ids[i] in self.ws_ids:
                i += 1

            # role 'assistant' 매칭
            j = i
            ok = True
            for rid in self.assistant_ids:
                if j >= T or ids[j] != rid:
                    ok = False
                    break
                j += 1
            # role 뒤 공백/개행 스킵
            while ok and j < T and ids[j] in self.ws_ids:
                j += 1

            # im_end 찾기
            k = j
            while k < T and ids[k] != self.im_end_id:
                k += 1
            if k >= T: break

            # assistant 세그먼트면, role 이후(j) ~ im_end 직전(k)까지 라벨 on
            if ok and j <= k:
                mask[j:k+1] = True  # k(=IM_END) 포함하여 EOS까지 학습

            # 다음 탐색은 im_end 다음부터
            i = k + 1
        return mask
    
    def __call__(self, features):
        feats = []
        for ex in features:
            ex = copy.deepcopy(ex)
            msgs = _clean_messages(ex.get("messages", []))

            if len(msgs) >= 2:
                user_parts = msgs[0]["content"]
                asst_parts = msgs[1]["content"]

                if self.require_user_image and self._count_images(user_parts) < 1:
                    continue
                if self.require_asst_text and not self._has_text(asst_parts):
                    continue
            
            ex["messages"] = msgs
            feats.append(ex)

        if len(feats) == 0:
            raise ValueError(
                "[FixMaskCollator] Cleaning/filtering 이후 유효 샘플이 0입니다. "
                "-> 이미지 경로 누락/손상 또는 assistant 텍스트 누락이 많은지 점검하세요."
            )
        
        batch = self.base(feats)
        labels = batch["input_ids"].clone()

        B, T = labels.size()
        keep_mask = torch.zeros_like(labels, dtype=torch.bool)
        for b in range(B):
            keep_mask[b] = self._assistant_mask_from_ids_tokenwise(batch["input_ids"][b])

        labels[~keep_mask] = -100

        # 비전 토큰은 항상 무시
        if self.image_token_ids:
            img_mask = torch.zeros_like(labels, dtype=torch.bool)
            for tid in self.image_token_ids:
                img_mask |= (batch["input_ids"] == tid)
            labels[img_mask] = -100

        batch["labels"] = labels

        # attention_mask dtype 정규화
        if "attention_mask" in batch and batch["attention_mask"].dtype == torch.long:
            batch["attention_mask"] = batch["attention_mask"].bool()
        return batch
    
data_collator = FixMaskCollator(model, processor, require_user_image=True, require_asst_text=True)

os.environ["WANDB_PROJECT"] = wandb_project
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_CACHE"] = "/workspace/hf_cache"
kv_table("ENV", {
    "WANDB_PROJECT": os.environ.get("WANDB_PROJECT"),
    "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
    "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
})

training_args = TrainingArguments(

    # --- 성능 및 메모리 관련 ---
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    gradient_checkpointing = True,
    bf16 = True,
    optim = "adamw_8bit",

    # --- 학습 파라미터 ---
    learning_rate = 2e-4,
    lr_scheduler_type = "cosine",
    weight_decay = 0.0,
    warmup_ratio = 0.05,
    max_grad_norm = 0.3,
    num_train_epochs = 2.0,
    remove_unused_columns = False,

    # --- 로깅 및 저장 ---
    logging_steps = 5,
    save_steps = 50,
    output_dir = "outputs_py",
    report_to = "wandb",
    run_name = wandb_run,
    save_total_limit = 3,
)

trainer_kwargs = dict(
    model = model,
    args = training_args,
    train_dataset = processed_dataset,
    data_collator = data_collator,
)

if _HAS_RICH_PROGRESS:
    trainer_kwargs["callbacks"] = [RichProgressCallback()]
else:
    warn("RichProgressCallback을 사용할 수 없어 기본 tqdm 진행바로 대체합니다. "
         "(transformers 버전 또는 extras 미설치 가능)")

trainer = Trainer(**trainer_kwargs)

banner("학습 환경 구성 완료. 학습 시작!", "🎯")
kv_table("TrainingArgs 요약", {
    "per_device_batch": training_args.per_device_train_batch_size,
    "grad_accum": training_args.gradient_accumulation_steps,
    "epochs": training_args.num_train_epochs,
    "save_steps": training_args.save_steps,
    "output_dir": training_args.output_dir,
})

try:
    trainer.train()
    ok("Training finished")
except Exception as e:
    fail(f"Training failed: {e}")

