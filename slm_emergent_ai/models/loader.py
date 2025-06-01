"""
モデルローダー - モデルの読み込みと量子化を行うモジュール

gemma3:1bモデルの読み込みと量子化を行うユーティリティ関数を提供。
"""

import os
import torch
from typing import Dict, Any, Optional

def load_model(model_path: str, quantize: str = "q4", threads: int = 4) -> Dict[str, Any]:
    """
    モデルを読み込む
    
    Parameters:
    -----------
    model_path: モデルパス
    quantize: 量子化レベル
    threads: スレッド数
    
    Returns:
    --------
    モデルとトークナイザーを含む辞書
    """
    try:
        # CPUで推論を行う設定
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPUを無効化
        os.environ["OMP_NUM_THREADS"] = str(threads)
        
        # モデルのロード
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        
        print(f"Model loaded with {threads} threads")
        
        return {
            "model": model,
            "tokenizer": tokenizer
        }
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def quantize_model(model: Any, quantize: str = "q4") -> Any:
    """
    モデルを量子化する
    
    Parameters:
    -----------
    model: モデル
    quantize: 量子化レベル
    
    Returns:
    --------
    量子化されたモデル
    """
    try:
        if quantize == "q4":
            # 4ビット量子化
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # モデルを再読み込み
            model_id = model.config._name_or_path
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="cpu"
            )
            
            print("Model quantized to 4-bit")
        elif quantize == "q8":
            # 8ビット量子化
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            
            # モデルを再読み込み
            model_id = model.config._name_or_path
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="cpu"
            )
            
            print("Model quantized to 8-bit")
        
        return model
    except Exception as e:
        print(f"Error quantizing model: {e}")
        return model  # 量子化に失敗した場合は元のモデルを返す
