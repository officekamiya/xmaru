# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms import HuggingFacePipeline
import logging
import time # 処理時間計測用
import os # ローカルパスの確認に必要

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_quantized_pipeline(hub_model_id: str):
    """
    指定されたモデルID (Hub上のID) を元に、ローカルパスを優先的に検索し、
    量子化設定を行って transformers.pipeline を作成して返す関数。
    """
    logger.info(f"モデル: {hub_model_id} のパイプラインを準備します。")

    # --- ローカルパスの確認 ---
    # "google/gemma-1.1-2b-it" -> "gemma-1.1-2b-it" のようにモデル名部分を取得
    model_name_only = hub_model_id.split('/')[-1]
    # ローカルの 'models' ディレクトリ内のパスを生成
    local_model_path = os.path.join("models", model_name_only)

    if os.path.isdir(local_model_path):
        logger.info(f"ローカルパス '{local_model_path}' が見つかりました。こちらを使用します。")
        model_id_to_load = local_model_path # ロードにはローカルパスを使用
    else:
        logger.warning(f"ローカルパス '{local_model_path}' が見つかりません。Hugging Face Hub ({hub_model_id}) からのロードを試みます。")
        model_id_to_load = hub_model_id # ロードにはHubのIDを使用

    # --- 1. 量子化設定 ---
    # (この部分は変更なし)
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 # 環境に合わせて torch.float16 も可
        )
        logger.info("4bit量子化設定 (nf4, bfloat16) を作成しました。")
    except ImportError:
        logger.error("bitsandbytes が見つかりません。'pip install bitsandbytes accelerate' を実行してください。")
        raise
    except Exception as e:
        logger.error(f"量子化設定中に予期せぬエラー: {e}")
        raise

    # --- 2. トークナイザーとモデルのロード ---
    try:
        start_time = time.time()
        logger.info(f"トークナイザーを読み込み中 ({model_id_to_load})...")
        # ★★★ 決定した model_id_to_load を使用 ★★★
        tokenizer = AutoTokenizer.from_pretrained(model_id_to_load)

        logger.info(f"モデルを読み込み中 ({model_id_to_load}, 量子化適用)...")
        if not torch.cuda.is_available():
            logger.warning("警告: CUDA (GPU) が利用できません。量子化はGPUでのみ効果的に機能します。")
            raise RuntimeError("量子化にはCUDA対応GPUが必要です。")
        else:
            device_map = "auto"
            torch_dtype = quantization_config.bnb_4bit_compute_dtype
            logger.info(f"GPU ({torch.cuda.get_device_name(0)}) を使用します。")

        # ★★★ 決定した model_id_to_load を使用 ★★★
        model = AutoModelForCausalLM.from_pretrained(
            model_id_to_load, # ローカルパスまたはHub ID
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        end_time = time.time()
        logger.info(f"モデルの読み込み完了 ({end_time - start_time:.2f}秒)")

    except Exception as e:
        # ★★★ エラーメッセージにもロードしようとしたIDを表示 ★★★
        logger.error(f"モデルまたはトークナイザーの読み込み中にエラー ({model_id_to_load}): {e}")
        raise

    # --- 3. transformers.pipeline の作成 ---
    # (この部分は変更なし)
    try:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            return_full_text=False,
            framework="pt"
        )
        logger.info("transformers パイプラインを作成しました。")
        return pipe
    except Exception as e:
        logger.error(f"transformers パイプライン作成中にエラー: {e}")
        raise

def main():
    # --- 使用するモデルIDを指定 (Hugging Face Hub上のID形式で指定) ---
    # このIDを元に、まず 'models/モデル名' のローカルパスを探します
    hub_model_id = "google/gemma-3-4b-it" # 例: これを指定すると 'models/gemma-1.1-2b-it' を探しに行く

    # もし 'models/my-custom-gemma' のような独自のパスを使いたい場合は、
    # hub_model_id = 'models/my-custom-gemma' のように直接ローカルパスを指定することも可能です。
    # その場合、load_quantized_pipeline 関数内のローカルパス探索ロジックはスキップされます。

    logger.info(f"目的のモデル: {hub_model_id}")

    try:
        # --- 量子化モデルのパイプラインをロード ---
        start_time = time.time()
        # ★★★ Hub ID を関数に渡す ★★★
        transformer_pipeline = load_quantized_pipeline(hub_model_id)
        end_time = time.time()
        logger.info(f"パイプライン準備完了 ({end_time - start_time:.2f}秒)")

        # --- LangChain の HuggingFacePipeline を初期化 ---
        llm = HuggingFacePipeline(pipeline=transformer_pipeline)
        logger.info("LangChain HuggingFacePipeline を初期化しました。")

        # --- LangChain経由でプロンプトを実行 ---
        prompt = "AIを使った農作物ランク判別システムの開発における注意点を3つ挙げてください。"
        logger.info(f"以下のプロンプトで推論を実行します:\n{prompt}")
        start_time = time.time()
        response = llm.invoke(prompt)
        end_time = time.time()
        logger.info(f"推論完了 ({end_time - start_time:.2f}秒)")

        # --- 結果を表示 ---
        print("\n--- AIの応答 ---")
        print(response)
        print("--- 応答終了 ---")

        # (おまけのChainを使った例は変更なし)

    except Exception as e:
        logger.error(f"メイン処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc() # 詳細なトレースバックを表示

if __name__ == "__main__":
    # 必要なライブラリがインストールされているか確認 (簡易チェック)
    try:
        import torch
        import transformers
        import langchain_community
        import bitsandbytes
        import sentencepiece
        import accelerate
        import os # osモジュールもチェックに追加 (念のため)
    except ImportError as e:
        print(f"エラー: 必要なライブラリが見つかりません ({e.name})。")
        print("依存関係を確認し、'pip install ...' を実行してください。")
        print("例: pip install torch transformers bitsandbytes accelerate langchain-community sentencepiece")
    else:
        # スクリプトと同じディレクトリに 'models' ディレクトリがなければ作成 (任意)
        if not os.path.isdir("models"):
             logger.info("'models' ディレクトリが見つからないため作成します。")
             try:
                  os.makedirs("models")
             except OSError as e:
                  logger.warning(f"'models' ディレクトリの作成に失敗しました: {e}")

        main()