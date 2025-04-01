# 必要なライブラリをインポート
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig
)
import os
import time
import traceback # エラー表示用にインポート

# ==================================================
# --- 設定項目 ---
# ==================================================

# --- モデル設定 ---
MODEL_NAME = "gemma-3-4b-it" # 使用するモデル名 (Hugging Face Hub またはローカル)
# ローカルパスを使う場合は、以下の models ディレクトリからのパスになるように調整
# 例: _model_name = "gemma-3-4b-it"; model_id_or_path = os.path.join("models", _model_name)
# ローカルパスが存在しない場合は Hugging Face Hub からのロードを試みる
_local_model_path = os.path.join("models", MODEL_NAME)
if os.path.isdir(_local_model_path):
    model_id_or_path = _local_model_path
    print(f"Using local model: {model_id_or_path}")
else:
    model_id_or_path = f"google/{MODEL_NAME}"
    print(f"Warning: Local model directory '{_local_model_path}' not found.")
    print(f"Falling back to Hugging Face Hub: {model_id_or_path}")

# --- 量子化設定 ---
USE_QUANTIZATION = False   # 量子化を使用するか (True / False)
QUANTIZATION_LEVEL = "4bit" # "4bit" または "8bit" (USE_QUANTIZATION=True の場合)

# --- 4bit量子化 詳細設定 (USE_QUANTIZATION=True かつ QUANTIZATION_LEVEL="4bit" の場合) ---
# 安定性を考慮し、 compute_dtype=float16, double_quant=False をデフォルトに設定
# 必要に応じて "bfloat16", "float32" や True に変更して試す
BNB_4BIT_COMPUTE_DTYPE_STR = "bfloat16" # 計算時のデータ型: "bfloat16", "float16", "float32"
BNB_4BIT_QUANT_TYPE = "nf4"          # 量子化タイプ: "nf4", "fp4"
BNB_4BIT_USE_DOUBLE_QUANT = True      # ダブル量子化を使用するか: True, False

# --- データ型 (精度) 設定 ---
# 量子化しない場合、または8bit量子化の場合のモデルの基本データ型
# 4bit量子化の場合は bnb_4bit_compute_dtype が計算精度として優先されることが多い
MODEL_DTYPE_STR = "bfloat16" # "bfloat16", "float16", "float32"

# --- 入力設定 ---
IMAGE_PATH = "medias/test2.jpg" # 処理したい画像のパス
USER_QUESTION = "この画像には何が写っていますか？ 詳細に説明してください。"

# --- 推論 (generate) 設定 ---
DO_SAMPLE = True       # サンプリングを行うか (True / False)
MAX_NEW_TOKENS = 200    # 生成する最大トークン数
# --- サンプリング用パラメータ (DO_SAMPLE=True の場合) ---
TEMPERATURE = 0.7     # 温度 (低いほど決定的、高いほど多様)
TOP_P = 0.95          # Top-p サンプリング
TOP_K = 64            # Top-k サンプリング

# ==================================================
# --- 設定に基づいた変数の準備 ---
# ==================================================

# データ型文字列を torch.dtype オブジェクトに変換
def get_torch_dtype(dtype_str):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str.lower(), None)

model_dtype = get_torch_dtype(MODEL_DTYPE_STR)
bnb_4bit_compute_dtype = get_torch_dtype(BNB_4BIT_COMPUTE_DTYPE_STR)

if model_dtype is None and not (USE_QUANTIZATION and QUANTIZATION_LEVEL == "4bit"):
    print(f"Warning: Invalid MODEL_DTYPE_STR '{MODEL_DTYPE_STR}'. Using default (float32).")
    model_dtype = torch.float32 # デフォルトは float32

if bnb_4bit_compute_dtype is None and USE_QUANTIZATION and QUANTIZATION_LEVEL == "4bit":
     print(f"Warning: Invalid BNB_4BIT_COMPUTE_DTYPE_STR '{BNB_4BIT_COMPUTE_DTYPE_STR}'. Using default (float32).")
     bnb_4bit_compute_dtype = torch.float32 # 4bit計算時のデフォルト

# 量子化設定 (BitsAndBytesConfig) の構築
quantization_config = None
if USE_QUANTIZATION:
    if QUANTIZATION_LEVEL == "4bit":
        if bnb_4bit_compute_dtype is None:
             print("Error: Valid compute dtype required for 4bit quantization.")
             exit()
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
            bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        )
        print(f"Using 4-bit quantization:")
        print(f"  Compute dtype: {BNB_4BIT_COMPUTE_DTYPE_STR}")
        print(f"  Quant type: {BNB_4BIT_QUANT_TYPE}")
        print(f"  Use double quant: {BNB_4BIT_USE_DOUBLE_QUANT}")
    elif QUANTIZATION_LEVEL == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Using 8-bit quantization.")
    else:
        print(f"Warning: Invalid QUANTIZATION_LEVEL '{QUANTIZATION_LEVEL}'. Disabling quantization.")
        USE_QUANTIZATION = False # 設定が無効なら量子化しない

if not USE_QUANTIZATION:
    print("Quantization disabled.")
    quantization_config = None # 明示的に None に

# 入力データのdtype決定
# 量子化しない場合/8bitの場合はMODEL_DTYPE_STRに従う
# 4bitの場合はBNB_4BIT_COMPUTE_DTYPE_STRに従う
if USE_QUANTIZATION and QUANTIZATION_LEVEL == "4bit":
    input_dtype = bnb_4bit_compute_dtype
    print(f"Input data dtype will be set to 4bit compute dtype: {BNB_4BIT_COMPUTE_DTYPE_STR}")
else:
    input_dtype = model_dtype
    print(f"Input data dtype will be set to model dtype: {MODEL_DTYPE_STR}")
# 安全策としてNoneの場合はfloat32にする
if input_dtype is None:
    print("Warning: Determined input dtype is None. Defaulting to float32.")
    input_dtype = torch.float32

# プロンプト作成
prompt = f"USER: <image>\n{USER_QUESTION}\nASSISTANT:"
print(f"Using prompt: \n{prompt}")

# ==================================================
# --- メイン処理 ---
# ==================================================

# --- デバイス設定 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu" and USE_QUANTIZATION:
    print("警告: 量子化は通常GPUでの使用が推奨されます。CPUでは非常に遅いか動作しない可能性があります。")

# --- モデル、トークナイザー、プロセッサーのロード ---
print(f"Loading model from: {model_id_or_path}...")
start_time = time.time()
try:
    # モデルロード時の引数準備
    model_load_kwargs = {
        "quantization_config": quantization_config,
        "device_map": "auto",
        "trust_remote_code": True
    }
    # 量子化しない場合、または8bit量子化の場合は torch_dtype を指定
    # 4bit量子化の場合は torch_dtype を指定しない方が bitsandbytes との競合を避けられる場合がある
    if not USE_QUANTIZATION or QUANTIZATION_LEVEL == "8bit":
        if model_dtype is not None:
             model_load_kwargs["torch_dtype"] = model_dtype
             print(f"Loading model with torch_dtype: {MODEL_DTYPE_STR}")
        else:
             print("Loading model with default torch_dtype (usually float32)")
    else: # 4bit量子化の場合
        print("Loading 4bit model without explicit torch_dtype (using compute_dtype for calculation)")


    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        **model_load_kwargs
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    processor = AutoProcessor.from_pretrained(model_id_or_path)

# ...(エラーハンドリング)...
except torch.cuda.OutOfMemoryError as e:
    print("\n" + "="*30); print(" エラー: モデルロード中にGPUメモリ不足。"); print("="*30); exit()
except Exception as e:
    print(f"モデル等のロード中にエラー: {e}"); traceback.print_exc(); exit()
load_time = time.time() - start_time
print(f"Model, tokenizer, and processor loaded in {load_time:.2f} seconds.")

# --- 画像の読み込み ---
print(f"Loading image: {IMAGE_PATH}...")
start_time = time.time()
try:
    image = Image.open(IMAGE_PATH)
    if image.mode != "RGB": image = image.convert("RGB")
except FileNotFoundError: print(f"エラー: 画像ファイルが見つかりません: {IMAGE_PATH}"); exit()
except Exception as e: print(f"エラー: 画像の読み込み中に問題: {e}"); exit()
load_image_time = time.time() - start_time
print(f"Image loaded in {load_image_time:.2f} seconds.")

# --- messages リストの作成 ---
print("Creating messages list...")
messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": USER_QUESTION}]}]
print("Messages list created.")

# --- 入力データの準備 ---
print("Processing inputs using processor.apply_chat_template...")
start_time = time.time()
try:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=input_dtype) # 決定された入力データ型を使用

    print(f"Processor output keys: {inputs.keys()}")
    print(f"Input tensor dtype: input_ids={inputs['input_ids'].dtype}, pixel_values={inputs['pixel_values'].dtype}")

except Exception as e:
    print(f"入力データの処理中にエラー: {e}"); traceback.print_exc(); exit()
process_input_time = time.time() - start_time
print(f"Inputs processed in {process_input_time:.2f} seconds.")

# --- 推論の実行 ---
print("Generating response using model.generate...")
start_time = time.time()
try:
    input_len = inputs["input_ids"].shape[-1]
    print(f"Input length (input_len): {input_len}")

    # generate に渡す引数を設定変数から構築
    generate_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE
    }
    if DO_SAMPLE:
        generate_kwargs["temperature"] = TEMPERATURE
        generate_kwargs["top_p"] = TOP_P
        generate_kwargs["top_k"] = TOP_K
    print(f"Generate kwargs: {generate_kwargs}") # 確認用

    with torch.inference_mode():
        generation_output = model.generate(
            **inputs,
            **generate_kwargs
        )
    print(f"Generated output shape: {generation_output.shape}")

# ...(エラーハンドリング)...
except torch.cuda.OutOfMemoryError as e:
    print("\n" + "="*30); print(" エラー: 推論中にGPUメモリ不足。"); print("="*30); exit()
except Exception as e:
    print(f"推論の実行中にエラー: {e}"); traceback.print_exc(); exit()
generation_time = time.time() - start_time
print(f"Response generated in {generation_time:.2f} seconds.")

# --- 結果のデコード ---
print("Decoding response...")
start_time = time.time()
try:
    generated_ids = generation_output[0]
    # print(f"Generated IDs length: {len(generated_ids)}") # デバッグ用
    response_ids = generated_ids[input_len:]
    # print(f"Response IDs length: {len(response_ids)}") # デバッグ用
    if len(response_ids) > 0:
        # print(f"Response IDs (first 20): {response_ids[:20]}") # デバッグ用
        decoded_response = processor.decode(response_ids, skip_special_tokens=True)
    else:
        print("警告: 応答部分のトークンIDが空です。"); decoded_response = ""
except Exception as e: print(f"応答のデコード中にエラー: {e}"); decoded_response = "エラー"
decode_time = time.time() - start_time
print(f"Response decoded in {decode_time:.2f} seconds.")

# ...(応答表示、時間表示)...
print("\n" + "="*30)
print(f" Gemma 3 ({'Quantized: ' + QUANTIZATION_LEVEL if USE_QUANTIZATION else 'Non-Quantized'}) / apply_chat_template からの応答:") # タイトルに設定反映
print("="*30)
print(decoded_response)
print("="*30 + "\n")
print(f"Total time excluding model loading: {load_image_time + process_input_time + generation_time + decode_time:.2f} seconds")