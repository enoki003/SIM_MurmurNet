"""
メインスクリプト - SLM Emergent AIの実行エントリーポイント

複数のSLMエージェントを協調させ、Boidsアルゴリズムによる創発知能を実証するシステムの
メインスクリプト。コマンドライン引数の処理、設定の読み込み、システムの初期化と実行を行う。
"""

import os
import sys
import asyncio
import argparse
import time
from typing import Dict, List, Any, Optional
import yaml
import torch
import numpy as np

from .agents.core import SLMAgent, LLM
from .memory.blackboard import BlackBoard
from .controller.meta import MetaController
from .eval.metrics import MetricsTracker
from .ui.dashboard import init_app
from .agents.rag import RAGAgent


async def run_simulation(config: Dict[str, Any]):
    """
    シミュレーションを実行
    
    Parameters:
    -----------
    config: 設定辞書
    """
    print("Initializing SLM Emergent AI...")
    
    # BlackBoardの初期化
    bb_mode = config.get("memory", {}).get("mode", "local")
    redis_url = config.get("memory", {}).get("redis_url", None)
    bb = BlackBoard(mode=bb_mode, redis_url=redis_url)
    
    # モデルの初期化
    model_path = config.get("model", {}).get("path", "gemma3:1b")
    threads = config.get("runtime", {}).get("threads", 4)
    quantize = config.get("model", {}).get("quantize", "q4")
    
    print(f"Loading model: {model_path} with {threads} threads, quantization: {quantize}")
    model = LLM(model_path, threads=threads, quantize=quantize)
    
    # エージェントの初期化
    agent_count = config.get("agent", {}).get("n", 3)
    agents = []
    
    # λパラメータの初期化
    lambda_params = {
        "λ_a": config.get("lambda", {}).get("a", 0.3),
        "λ_c": config.get("lambda", {}).get("c", 0.3),
        "λ_s": config.get("lambda", {}).get("s", 0.1)
    }
    
    # BlackBoardにλパラメータを設定
    for key, value in lambda_params.items():
        await bb.set_param(key, value)
    
    # 基準速度を設定
    await bb.set_param("base_speed", 10.0)
    
    print(f"Initializing {agent_count} agents...")
    for i in range(agent_count):
        role = f"Agent{i+1}"
        agent = SLMAgent(id=i+1, role=role, model=model, tokenizer=model.tokenizer, λ=lambda_params)
        agents.append(agent)
    
    # RAGエージェントの初期化（オプション）
    rag_enabled = config.get("rag", {}).get("enabled", False)
    rag_agent = None
    if rag_enabled:
        backend_type = config.get("rag", {}).get("backend", "chroma")
        backend_config = config.get("rag", {}).get("config", {})
        print(f"Initializing RAG with {backend_type} backend...")
        rag_agent = RAGAgent(backend_type=backend_type, backend_config=backend_config, model_path=model_path)
    
    # MetaControllerの初期化
    target_H = config.get("controller", {}).get("target_H", 5.0)
    Kp = config.get("controller", {}).get("Kp", 0.1)
    Ki = config.get("controller", {}).get("Ki", 0.01)
    Kd = config.get("controller", {}).get("Kd", 0.05)
    controller = MetaController(bb=bb, target_H=target_H, Kp=Kp, Ki=Ki, Kd=Kd)
    
    # メトリクストラッカーの初期化
    metrics_tracker = MetricsTracker()
    
    # UIダッシュボードの初期化
    ui_enabled = config.get("ui", False)
    if ui_enabled:
        ui_port = config.get("ui", {}).get("port", 7860)
        app = init_app(blackboard=bb)
        
        # UIを別スレッドで起動
        import uvicorn
        import threading
        ui_thread = threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={"host": "0.0.0.0", "port": ui_port},
            daemon=True
        )
        ui_thread.start()
        print(f"UI dashboard started on port {ui_port}")
    
    # シミュレーションの実行
    print("Starting simulation...")
    
    # 初期プロンプトの設定
    initial_prompts = config.get("prompts", {}).get("initial", ["こんにちは、私はAIアシスタントです。"])
    
    # コントローラーを別タスクで実行
    controller_task = asyncio.create_task(controller.run())
    
    # エージェントの実行
    agent_tasks = []
    for i, agent in enumerate(agents):
        prompt = initial_prompts[i % len(initial_prompts)]
        # max_turns を非常に大きな値に設定して、実質的に時間制限なしにする
        task = asyncio.create_task(agent.run_conversation(prompt, bb, max_turns=sys.maxsize)) # Pythonの整数型の最大値
        agent_tasks.append(task)
    
    try:
        # すべてのエージェントタスクが完了するのを待つ
        await asyncio.gather(*agent_tasks)
    except KeyboardInterrupt:
        print("\\nCtrl+Cが押されました。シミュレーションを安全に停止します...")
    finally:
        # コントローラーを停止
        if controller_task and not controller_task.done():
            controller_task.cancel()
            try:
                await controller_task
            except asyncio.CancelledError:
                print("コントローラータスクがキャンセルされました。")
        
        # エージェントタスクもキャンセル (必要に応じて)
        for task in agent_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass # エージェントタスクのキャンセルは想定内
        
        print("クリーンアップ完了。")

    print("Simulation completed!")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SLM Emergent AI")
    parser.add_argument("-c", "--config", type=str, default="configs/base.yml", help="設定ファイルのパス")
    parser.add_argument("-m", "--multi", action="store_true", help="複数の設定を並列実行")
    parser.add_argument("--agent.n", type=int, help="エージェント数")
    parser.add_argument("--model.path", type=str, help="モデルパス")
    parser.add_argument("--runtime.threads", type=int, help="スレッド数")
    parser.add_argument("--ui", type=bool, help="UIの有効化")
    
    args = parser.parse_args()
    
    # 設定ファイルの読み込み
    if os.path.isabs(args.config):
        config_path = args.config
    else:
        # 相対パスの場合、slm_emergent_aiパッケージディレクトリからの相対パスとして扱う
        package_dir = os.path.dirname(__file__)
        config_path = os.path.join(package_dir, args.config)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # コマンドライン引数で設定を上書き
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name != "config" and arg_name != "multi":
            # ドット区切りの引数を階層的に設定
            keys = arg_name.split(".")
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = arg_value
    
    # シミュレーションの実行
    asyncio.run(run_simulation(config))


if __name__ == "__main__":
    main()
