"""
モデル量子化モジュール - モデルの量子化を行うユーティリティ

gemma3:1bモデルなどの量子化を行うユーティリティ関数を提供。
int4, int8, float16などの量子化をサポート。
"""

import os
import torch
from typing import Dict, Any, Optional, Union, Tuple

def quantize_model_to_int4(model: Any) -> Any:
    """
    モデルをint4形式に量子化する
    
    Parameters:
    -----------
    model: 量子化するモデル
    
    Returns:
    --------
    量子化されたモデル
    """
    try:
        from transformers import BitsAndBytesConfig
        
        # 4ビット量子化の設定
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",  # 正規化浮動小数点形式
            bnb_4bit_use_double_quant=True  # 二重量子化で更にメモリ効率を向上
        )
        
        # モデルを再読み込み
        model_id = model.config._name_or_path
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="cpu"
        )
        
        print(f"Model quantized to int4 format")
        return model
    except Exception as e:
        print(f"Error quantizing model to int4: {e}")
        return model

def quantize_model_to_int8(model: Any) -> Any:
    """
    モデルをint8形式に量子化する
    
    Parameters:
    -----------
    model: 量子化するモデル
    
    Returns:
    --------
    量子化されたモデル
    """
    try:
        from transformers import BitsAndBytesConfig
        
        # 8ビット量子化の設定
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        # モデルを再読み込み
        model_id = model.config._name_or_path
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="cpu"
        )
        
        print(f"Model quantized to int8 format")
        return model
    except Exception as e:
        print(f"Error quantizing model to int8: {e}")
        return model

def quantize_model_to_float16(model: Any) -> Any:
    """
    モデルをfloat16形式に量子化する
    
    Parameters:
    -----------
    model: 量子化するモデル
    
    Returns:
    --------
    量子化されたモデル
    """
    try:
        # float16に変換
        model = model.half()
        
        print(f"Model quantized to float16 format")
        return model
    except Exception as e:
        print(f"Error quantizing model to float16: {e}")
        return model

def quantize_model(model: Any, quantize_type: str = "q4") -> Any:
    """
    モデルを指定された形式に量子化する
    
    Parameters:
    -----------
    model: 量子化するモデル
    quantize_type: 量子化タイプ ("q4", "q8", "f16")
    
    Returns:
    --------
    量子化されたモデル
    """
    if quantize_type == "q4":
        return quantize_model_to_int4(model)
    elif quantize_type == "q8":
        return quantize_model_to_int8(model)
    elif quantize_type == "f16":
        return quantize_model_to_float16(model)
    else:
        print(f"Unknown quantization type: {quantize_type}, using original model")
        return model

def save_quantized_model(model: Any, tokenizer: Any, output_path: str) -> bool:
    """
    量子化されたモデルを保存する
    
    Parameters:
    -----------
    model: 量子化されたモデル
    tokenizer: トークナイザー
    output_path: 出力パス
    
    Returns:
    --------
    成功したかどうか
    """
    try:
        os.makedirs(output_path, exist_ok=True)
        
        # モデルとトークナイザーを保存
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        # 量子化情報を保存
        with open(os.path.join(output_path, "quantization_info.txt"), "w") as f:
            f.write(f"Model type: {model.config.model_type}\n")
            f.write(f"Quantization: {model.config._name_or_path}\n")
            if hasattr(model.config, "quantization_config"):
                f.write(f"Quantization config: {model.config.quantization_config}\n")
        
        print(f"Quantized model saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving quantized model: {e}")
        return False

def load_and_quantize_model(model_path: str, quantize_type: str = "q4", output_path: Optional[str] = None) -> Tuple[Any, Any]:
    """
    モデルを読み込んで量子化する
    
    Parameters:
    -----------
    model_path: モデルパス
    quantize_type: 量子化タイプ
    output_path: 出力パス (Noneの場合は保存しない)
    
    Returns:
    --------
    (量子化されたモデル, トークナイザー)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # トークナイザーを読み込む
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # モデルを読み込む
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        
        # モデルを量子化
        quantized_model = quantize_model(model, quantize_type)
        
        # 出力パスが指定されている場合は保存
        if output_path:
            save_quantized_model(quantized_model, tokenizer, output_path)
        
        return quantized_model, tokenizer
    except Exception as e:
        print(f"Error loading and quantizing model: {e}")
        raise
