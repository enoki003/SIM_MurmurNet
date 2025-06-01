#!/bin/bash
# モデル取得スクリプト

MODEL_NAME=$1
OUTPUT_DIR="models"

if [ -z "$MODEL_NAME" ]; then
  echo "使用法: $0 <model_name>"
  echo "例: $0 gemma3:1b"
  exit 1
fi

# 出力ディレクトリの作成
mkdir -p $OUTPUT_DIR

echo "モデル $MODEL_NAME をダウンロードしています..."

if [ "$MODEL_NAME" == "gemma3:1b" ]; then
  # Gemma 3 1B モデルのダウンロード
  python -m pip install -q huggingface_hub
  python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-3-1b-it-qat-q4_0-gguf', local_dir='$OUTPUT_DIR/gemma3-1b')"
  echo "モデルを $OUTPUT_DIR/gemma3-1b に保存しました"
elif [ "$MODEL_NAME" == "gpt2" ]; then
  # GPT-2 モデルのダウンロード
  python -m pip install -q huggingface_hub
  python -c "from huggingface_hub import snapshot_download; snapshot_download('gpt2', local_dir='$OUTPUT_DIR/gpt2')"
  echo "モデルを $OUTPUT_DIR/gpt2 に保存しました"
else
  # その他のモデルはHugging Faceからダウンロード
  python -m pip install -q huggingface_hub
  python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_NAME', local_dir='$OUTPUT_DIR/$MODEL_NAME')"
  echo "モデルを $OUTPUT_DIR/$MODEL_NAME に保存しました"
fi

echo "ダウンロード完了!"
