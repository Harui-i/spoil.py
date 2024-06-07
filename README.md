# spoil.py
download test cases, generate solution code using OpenAI API, if the code passes the test, submit it

## 使いかた
1. online-judge toolsをインストールして、`oj submit`などができるようにする。
2. asyncioも必要かも。
3. `OJ_COOKIE_JAR_PATH`, `OPENAI_API_KEY`などを設定する。(bashなら ~/.bashrcに、 `export OJ_COOKIE_PATH='HOGEHOGE'` などと書く)
4. `python3 spoil.py abc355` などとする
## TODO
Geminiへの対応をやろうとしたものの、間に合いませんでした。

## 注意
AtCoderのルール変更などで、こういったプログラムの使用が今後禁止される可能性もあります。自分の責任のもとで使ってください。
→ルール変更により、コンテスト時間中のABCでの使用は禁止されました。　https://info.atcoder.jp/entry/llm-abc-rules-ja