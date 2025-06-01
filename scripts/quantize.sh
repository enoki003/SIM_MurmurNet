#!/bin/bash
# モデル量子化スクリプト

MODEL_NAME=$1
QUANT_TYPE=$2
OUTPUT_DIR="models"

if [ -z "$MODEL_NAME" ] || [ -z "$QUANT_TYPE" ]; then
  echo "使用法: $0 <model_name> <quant_type>"
  echo "例: $0 gpt2 q4"
  echo "量子化タイプ: q4, q8"
  exit 1
fi

# 出力ディレクトリの確認
if [ ! -d "$OUTPUT_DIR/$MODEL_NAME" ]; then
  echo "エラー: モデル $MODEL_NAME が $OUTPUT_DIR に存在しません"
  echo "先に ./dl_model.sh $MODEL_NAME を実行してください"
  exit 1
fi

echo "モデル $MODEL_NAME を $QUANT_TYPE 形式に量子化しています..."

# 量子化スクリプトの実行
python -c "
import os
import sys
sys.path.append('$(pwd)')
from slm_emergent_ai.models.loader import load_model, quantize_model
import torch

# モデルを読み込む
model_dict = load_model('$OUTPUT_DIR/$MODEL_NAME')
model = model_dict['model']
tokenizer = model_dict['tokenizer']

# モデルを量子化
quantized_model = quantize_model(model, '$QUANT_TYPE')

# 量子化モデルを保存
output_path = '$OUTPUT_DIR/${MODEL_NAME}-${QUANT_TYPE}'
os.makedirs(output_path, exist_ok=True)
quantized_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f'量子化モデルを {output_path} に保存しました')
"

if [ $? -eq 0 ]; then
  echo "量子化完了!"
else
  echo "量子化に失敗しました"
  exit 1
fi
