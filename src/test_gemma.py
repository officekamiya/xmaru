# pipeline_quant_test.py (コメント追記例)
import torch
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer
import os

# --- 設定項目 ---

# ▼▼▼ モデルIDの指定 ▼▼▼
# Hugging Face Hub のモデル名、またはローカルにダウンロードしたモデルのパスを指定
# --- Hugging Face Hub 上の Gemma 3 モデル例 ---
# google/gemma-3-4b-it  (4B, Instruction-tuned) <- 現在使用中
# google/gemma-3-4b-pt  (4B, Pre-trained)
# google/gemma-3-1b-it  (1B, Instruction-tuned)
# google/gemma-3-1b-pt  (1B, Pre-trained)
# google/gemma-3-12b-it (12B, Instruction-tuned)
# google/gemma-3-12b-pt (12B, Pre-trained)
# google/gemma-3-27b-it (27B, Instruction-tuned)
# google/gemma-3-27b-pt (27B, Pre-trained)
# ※モデルサイズが大きいほど高性能だが、多くのメモリ(VRAM)が必要
# ------------------------------------------
# model_id = "google/gemma-3-4b-it" # Hugging Face Hub から直接ロードする場合

# --- ローカルパスの指定 (推奨) ---
# 上記モデルを download_model.py などで事前にローカルにダウンロードした場合
_model_name = "gemma-3-4b-it" # ベースとなるモデル名 (上のリストから選択)
model_id = os.path.join("models", _model_name) # ローカルパス (例: "models/gemma-3-4b-it")
if not os.path.isdir(model_id):
    print(f"Warning: Local model directory '{model_id}' not found. Falling back to Hugging Face Hub.")
    model_id = f"google/{_model_name}" # 見つからない場合はHubからロード試行
# ------------------------------------------

# ▼▼▼ 量子化設定 (BitsAndBytesConfig) ▼▼▼
# 量子化はモデルの重みをより少ないビット数で表現し、メモリ使用量を削減、速度を向上させる技術。
# bitsandbytesライブラリを使った動的な量子化を行うか設定する。

use_quantization = True  # 量子化を使用するかどうか (True/False)
quantization_level = "4bit" # "4bit" または "8bit" を指定 (use_quantization=True の場合)

if use_quantization:
    if quantization_level == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # --- 4bit量子化の詳細設定 ---
            bnb_4bit_compute_dtype=torch.bfloat16, # 計算時のデータ型 (torch.bfloat16, torch.float16, torch.float32)
                                                   # 量子化された重みを計算時に一時的にどの型に変換するか。
            bnb_4bit_quant_type="nf4",             # 量子化の方式 ("nf4" or "fp4")
                                                   # "nf4" (NormalFloat 4-bit) が推奨されることが多い。
            bnb_4bit_use_double_quant=True,        # ダブル量子化を使うか (True/False)
                                                   # メモリ効率をさらに改善する場合がある。
        )
    elif quantization_level == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # --- 8bit量子化の詳細設定 (INT8) ---
            # llm_int8_threshold=6.0,             # 外れ値検出の閾値
            # llm_int8_enable_fp32_cpu_offload=False, # FP32重みをCPUにオフロードするか
            # llm_int8_has_fp16_weight=False,     # モデルの重みがFP16形式か
            # llm_int8_skip_modules=None,         # 量子化をスキップするモジュール名のリスト
        )
    else:
        print(f"Warning: Invalid quantization_level '{quantization_level}'. Disabling quantization.")
        quantization_config = None
        use_quantization = False # 無効なレベルの場合は量子化を無効にする

else:
    quantization_config = None # 量子化しない場合

print(f"Loading model: {model_id}")
print(f"Using quantization: {use_quantization}")
if use_quantization and quantization_config:
    print(f"Quantization level: {quantization_level}")
    print(f"Quantization config: {quantization_config.to_dict()}") # .to_dict() で見やすく表示

# ▼▼▼ モデルロード時/計算時の基本データ型 (torch_dtype) ▼▼▼
# モデルの重みをロードする際や、量子化されていない部分の計算、
# または量子化された重みを計算時に展開する際の基本的なデータ型を指定。
# 量子化設定(compute_dtype)と合わせてVRAM使用量、速度、精度に影響する。
# None の場合は torch.float32 (単精度) になる。
# 量子化しない場合は、ここで精度 (32bit/BF16/FP16) を選択する。

# --- データ型の選択肢 ---
# torch.float32 (None): 32bit単精度。最も精度が高いがメモリ使用量大。
# torch.bfloat16:       16bit半精度。Ampere世代以降のGPUで高速。FP16より扱いやすいとされる。
# torch.float16:        16bit半精度。多くのGPUで利用可能だが、扱える数値範囲が狭い。
# ----------------------
model_dtype = torch.bfloat16 # 現在の設定 (BF16)

print(f"Model default dtype: {model_dtype}")

# --- ここまで設定項目 ---

try:
    # トークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # pipelineの準備
    pipe = pipeline(
        "text-generation", # または "image-text-to-text"
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={
            "quantization_config": quantization_config, # 上で設定した量子化Configを渡す
            "device_map": "auto", # 自動的にGPUを割り当て
            "torch_dtype": model_dtype, # 上で設定したデータ型を渡す
            # "attn_implementation": "flash_attention_2", # 対応GPUなら高速化の可能性 (要別途インストール)
        }
    )

    print("Pipeline created successfully!")

    # 簡単なテキスト生成テスト
    prompt = "農作物のランク判別について、AIができることを3つ教えてください。"
    print("\nGenerating text...")
    outputs = pipe(prompt, max_new_tokens=150, do_sample=False)

    print("\n--- Generated Text ---")
    print(outputs[0]['generated_text'])
    print("--- End of Text ---")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    import traceback
    traceback.print_exc()
    print("\nモデルのロードや推論中にエラーが発生しました。")