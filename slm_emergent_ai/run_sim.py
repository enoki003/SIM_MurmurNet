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
    bb = BlackBoard(mode=bb_mode, redis_url=redis_url)    # モデルの初期化
    model_path = config.get("model", {}).get("path", "gemma3:1b")
    threads = config.get("runtime", {}).get("threads", 4)
    quantize = config.get("model", {}).get("quantize", "q4")
    n_ctx = config.get("model", {}).get("n_ctx", 512)
    
    print(f"Loading model: {model_path} with {threads} threads, quantization: {quantize}, context length: {n_ctx}")
    
    try:
        model = LLM(model_path, threads=threads, quantize=quantize, n_ctx=n_ctx)
        print("Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"ERROR: Model file not found: {e}")
        print("Please ensure the model file exists at the specified path.")
        return
    except ImportError as e:
        print(f"ERROR: Required library not installed: {e}")
        print("Please install the required dependencies.")
        return
    except Exception as e:
        print(f"ERROR: Failed to initialize model: {e}")
        print("Please check your model configuration and try again.")
        return
    
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
    # 役割の事前定義
    roles = config.get("agent", {}).get("roles", ["質問者", "回答者", "批評者"])
    for i in range(agent_count):
        role = roles[i % len(roles)]  # 循環的に役割を割り当て
        agent = SLMAgent(id=i+1, role=role, model=model, tokenizer=model.tokenizer, λ=lambda_params)
        agents.append(agent)
        print(f"[DEBUG] Agent {i+1} initialized with role: {role}")
    
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
    
    # ログとメトリクス収集のための変数
    simulation_logs = []
    step_counter = 0
    
    # 評価設定
    eval_enabled = config.get("evaluation", {}).get("enabled", True)
    eval_freq = config.get("evaluation", {}).get("freq", 5)
    
    print(f"[DEBUG] Evaluation enabled: {eval_enabled}, frequency: {eval_freq}")
      # 初期プロンプトの設定
    initial_prompts = config.get("prompts", {}).get("initial", ["こんにちは、私はAIアシスタントです。"])
    
    # コントローラーを別タスクで実行
    controller_task = asyncio.create_task(controller.run())
    
    # メトリクス更新タスクを追加
    async def metrics_update_loop():
        """定期的にメトリクスを更新するループ"""
        nonlocal step_counter, simulation_logs
        while True:
            await asyncio.sleep(2)  # 2秒間隔でメトリクス更新
            step_counter += 1
              # BlackBoardから最新のメッセージを取得（辞書形式）
            recent_messages = await bb.pull_messages_raw(k=20)
            
            # ログに追加
            step_log = {
                'step': step_counter,
                'timestamp': __import__('time').strftime("%H:%M:%S"),
                'agent_responses': {msg.get('agent_id', 0): msg.get('text', '') for msg in recent_messages[-3:] if isinstance(msg, dict)},
                'total_messages': len(recent_messages)
            }
            simulation_logs.append(step_log)
            
            # 評価指標の計算
            if eval_enabled and (step_counter % eval_freq == 0):
                try:                    # メトリクスの計算
                    current_metrics = {
                        'step': step_counter,
                        'total_messages': len(recent_messages),
                        'unique_agents': len(set(msg.get('agent_id', 0) for msg in recent_messages if isinstance(msg, dict))),
                        'avg_message_length': sum(len(msg.get('text', '')) for msg in recent_messages[-10:] if isinstance(msg, dict)) / min(10, len(recent_messages)) if recent_messages else 0,
                        'conversation_diversity': len(set(msg.get('text', '') for msg in recent_messages[-10:] if isinstance(msg, dict))) / min(10, len(recent_messages)) if recent_messages else 0
                    }
                    
                    print(f"[METRICS] Step {step_counter}: {current_metrics}")
                    
                    # メトリクストラッカーに記録
                    metrics_tracker.log_step(current_metrics)
                    
                except Exception as e:
                    print(f"[ERROR] Metrics calculation failed: {e}")
    
    # メトリクス更新タスクを開始
    metrics_task = asyncio.create_task(metrics_update_loop())
    
    # エージェントの実行
    agent_tasks = []
    for i, agent in enumerate(agents):
        prompt = initial_prompts[i % len(initial_prompts)]
        # max_turns を非常に大きな値に設定して、実質的に時間制限なしにする
        task = asyncio.create_task(agent.run_conversation(prompt, bb, max_turns=sys.maxsize)) # Pythonの整数型の最大値
        agent_tasks.append(task)
    try:
        # すべてのタスクを並行実行
        await asyncio.gather(*agent_tasks, metrics_task)
    except KeyboardInterrupt:
        print("\\nCtrl+Cが押されました。シミュレーションを安全に停止します...")
    finally:
        # すべてのタスクを停止
        tasks_to_cancel = [controller_task, metrics_task] + agent_tasks
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        print("クリーンアップ完了。")
        print(f"[SUMMARY] Total simulation steps: {step_counter}, Total logs: {len(simulation_logs)}")

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
        # 相対パスの場合、まずslm_emergent_aiディレクトリからの相対パスを試す
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, args.config)
        
        # ファイルが存在しない場合は、現在の作業ディレクトリからの相対パスを試す
        if not os.path.exists(config_path):
            config_path = os.path.abspath(args.config)
    
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
