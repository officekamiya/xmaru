# test_gemini_langchain.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    # 環境変数に API キーを設定済みと仮定
    # export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"

    # langchain-google-genai が用意しているクラス
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # 例：最新のモデル名を指定
        temperature=0.7,
        top_p=0.9,
    )

    # .invoke() で問い合わせ
    response = llm.invoke("こんにちは。あなたにはLangchainを介してGemini APIを使って問い合わせているはずです。それって判別できるのでしょうか？")
    print("Response:", response)

if __name__ == "__main__":
    main()
