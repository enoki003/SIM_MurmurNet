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
from .ui.dashboard import Dashboard
from .agents.rag import RAGAgent


async def agent_continuous_run(agent, initial_prompt, bb, restart_interval, max_runtime_hours, start_time):
    """
    エージェントを連続実行するための関数（エラーハンドリング強化版）
    """
    cycle_count = 0
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while True:
        # 時間制限チェック
        if max_runtime_hours:
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours >= max_runtime_hours:
                print(f"Agent {agent.id} stopping after {elapsed_hours:.2f} hours")
                break
        
        # 連続エラーが多い場合は停止
        if consecutive_errors >= max_consecutive_errors:
            print(f"Agent {agent.id} stopped due to {consecutive_errors} consecutive errors")
            break
        
        try:
            cycle_count += 1
            print(f"Agent {agent.id} starting cycle {cycle_count}")
            
            # モデル推論中のKeyboardInterrupt対策
            try:
                print(f"Agent {agent.id} calling run_conversation...")
                conversation_result = await agent.run_conversation(initial_prompt, bb, max_turns=restart_interval)
                print(f"Agent {agent.id} completed conversation with {len(conversation_result)} messages")
                consecutive_errors = 0  # 成功したらエラーカウンターリセット
            except KeyboardInterrupt:
                print(f"Agent {agent.id} interrupted by user. Gracefully shutting down...")
                break
            except Exception as model_error:
                print(f"Model inference error for agent {agent.id}: {model_error}")
                import traceback
                traceback.print_exc()
                consecutive_errors += 1
                await asyncio.sleep(3)  # モデルエラー時は短めに待機
                continue
            
            # 短い休憩
            await asyncio.sleep(1)
            
        except KeyboardInterrupt:
            print(f"Agent {agent.id} received keyboard interrupt. Shutting down gracefully...")
            break
        except Exception as e:
            print(f"Agent {agent.id} error in cycle {cycle_count}: {e}")
            consecutive_errors += 1
            await asyncio.sleep(5)  # エラー時は少し長く待機


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
    ui_config = config.get("ui", {})
    ui_enabled = ui_config.get("enabled", False) if isinstance(ui_config, dict) else False
    dashboard = None
    if ui_enabled:
        ui_port = ui_config.get("port", 7860) if isinstance(ui_config, dict) else 7860
        dashboard = Dashboard(port=ui_port, bb=bb)
        
        # UIを別スレッドで起動
        import uvicorn
        import threading
        ui_thread = threading.Thread(
            target=uvicorn.run,
            args=(dashboard.app,),
            kwargs={"host": "0.0.0.0", "port": ui_port},
            daemon=True
        )
        ui_thread.start()
        print(f"UI dashboard started on port {ui_port}")
    
    # メトリクス更新機能を追加
    async def update_metrics():
        """定期的にメトリクスを計算・更新"""
        while True:
            try:
                # BlackBoardから最新データを取得
                messages = await bb.pull(k=100)
                
                if messages:
                    print(f"[METRICS DEBUG] Processing {len(messages)} messages for metrics calculation")
                    
                    # エントロピー計算
                    text_lengths = [len(str(msg)) for msg in messages]
                    if text_lengths:
                        avg_length = sum(text_lengths) / len(text_lengths)
                        variance = sum((x - avg_length) ** 2 for x in text_lengths) / len(text_lengths)
                        entropy = min(max(variance / 100.0, 0.0), 10.0)
                        
                        # VDI計算（多様性指標）
                        unique_prefixes = set(str(msg)[:50] for msg in messages)
                        vdi = len(unique_prefixes) / max(len(messages), 1)
                        
                        # FCR計算（フロー継続率）
                        fcr = min(len(messages) / 50.0, 1.0)
                        
                        # 速度計算
                        current_speed = await bb.get_param("base_speed") or 14.0
                        
                        # メトリクスをBlackBoardに保存
                        await bb.set_param("current_entropy", entropy)
                        await bb.set_param("current_vdi", vdi)
                        await bb.set_param("current_fcr", fcr)
                        await bb.set_param("current_speed", current_speed)
                        
                        print(f"[METRICS DEBUG] Metrics saved to BlackBoard - Entropy: {entropy:.2f}, VDI: {vdi:.2f}, FCR: {fcr:.2f}, Speed: {current_speed:.1f}")
                        
                        # 検証: 保存されたデータを再読み込み
                        verify_entropy = await bb.get_param("current_entropy")
                        verify_vdi = await bb.get_param("current_vdi")
                        print(f"[METRICS DEBUG] Verification - Retrieved from BlackBoard: entropy={verify_entropy}, vdi={verify_vdi}")
                        
                else:
                    print("[METRICS DEBUG] No messages found for metrics calculation")
                
                await asyncio.sleep(3)  # 3秒間隔で更新
                
            except Exception as e:
                print(f"[METRICS DEBUG] Error updating metrics: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(3)
    
    # シミュレーションの実行
    print("Starting simulation...")
    
    # 初期プロンプトの設定
    initial_prompts = config.get("prompts", {}).get("initial", [
        "こんにちは、私はAIアシスタントです。今日はどのようなお手伝いができますか？",
        "AIについて議論しましょう。",
        "創発知能システムについて考えてみましょう。"
    ])
    
    # コントローラーを別タスクで実行
    controller_task = asyncio.create_task(controller.run())
    
    # メトリクス更新タスクを開始
    metrics_task = asyncio.create_task(update_metrics())
    
    # エージェントの実行
    agent_tasks = []
    max_turns = config.get("agent", {}).get("max_turns", 5)
    continuous_mode = config.get("agent", {}).get("continuous", False)
    restart_interval = config.get("agent", {}).get("restart_interval", 100)
    max_runtime_hours = config.get("simulation", {}).get("max_runtime_hours", None)
    
    # 長時間実行の場合の時間制限設定
    start_time = time.time()
    
    for i, agent in enumerate(agents):
        prompt = initial_prompts[i % len(initial_prompts)]
        print(f"Starting Agent {i+1} ({agent.role}) with prompt: '{prompt}'")
        
        if continuous_mode or max_turns == -1:
            # 連続実行モード
            if max_runtime_hours:
                print(f"Running in continuous mode for max {max_runtime_hours} hours")
            else:
                print("Running in infinite continuous mode")
            task = asyncio.create_task(
                agent_continuous_run(agent, prompt, bb, restart_interval, max_runtime_hours, start_time)
            )
        else:
            # 通常モード
            task = asyncio.create_task(agent.run_conversation(prompt, bb, max_turns=max_turns))
        agent_tasks.append(task)
    
    # 定期的なステータス更新
    async def status_updater():
        while True:
            await asyncio.sleep(10)  # 10秒間隔で更新
            stats = await bb.get_stats()
            print(f"BlackBoard Stats: {stats['total_messages']} messages, {len(stats['active_agents'])} active agents")
    
    status_task = asyncio.create_task(status_updater())
    
    # すべてのエージェントタスクが完了するのを待つ（エラーハンドリング付き）
    try:
        await asyncio.gather(*agent_tasks)
    except KeyboardInterrupt:
        print("\nシミュレーションが中断されました。タスクを停止中...")
        # 全てのタスクをキャンセル
        for task in agent_tasks:
            task.cancel()
        controller_task.cancel()
        metrics_task.cancel()
        status_task.cancel()
        
        # タスクの完了を待つ
        await asyncio.gather(*agent_tasks, return_exceptions=True)
        await asyncio.gather(controller_task, metrics_task, status_task, return_exceptions=True)
        print("すべてのタスクが停止されました。")
        return
    except Exception as e:
        print(f"シミュレーション実行中にエラーが発生: {e}")
        # エラー時もタスクをクリーンアップ
        for task in agent_tasks:
            task.cancel()
        controller_task.cancel()
        metrics_task.cancel()
        status_task.cancel()
        raise
    
    # タスクを停止
    controller_task.cancel()
    metrics_task.cancel()
    status_task.cancel()
    
    print("Simulation completed!")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SLM Emergent AI")
    parser.add_argument("-c", "--config", type=str, default="base.yml", help="設定ファイルのパス")
    parser.add_argument("-m", "--multi", action="store_true", help="複数の設定を並列実行")
    parser.add_argument("--agent.n", type=int, help="エージェント数")
    parser.add_argument("--model.path", type=str, help="モデルパス")
    parser.add_argument("--runtime.threads", type=int, help="スレッド数")
    parser.add_argument("--ui", action="store_true", help="UIの有効化")
    
    args = parser.parse_args()
    
    # 設定ファイルの読み込み
    if os.path.isabs(args.config) or "/" in args.config or "\\" in args.config:
        # 絶対パスまたは相対パスが指定された場合
        config_path = args.config
    else:
        # ファイル名のみが指定された場合、configsディレクトリを参照
        config_path = os.path.join(os.path.dirname(__file__), "configs", args.config)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # コマンドライン引数で設定を上書き
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name != "config" and arg_name != "multi":
            # UIフラグの特別処理
            if arg_name == "ui" and arg_value:
                # UIが有効化された場合、ui.enabledをTrueに設定
                if "ui" not in config:
                    config["ui"] = {}
                config["ui"]["enabled"] = True
                continue
                
            # ドット区切りの引数を階層的に設定
            keys = arg_name.split(".")
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = arg_value
    
    # シミュレーションの実行
    try:
        asyncio.run(run_simulation(config))
    except KeyboardInterrupt:
        print("\nシミュレーションがユーザーによって中断されました。")
        print("安全にシャットダウンしています...")
    except Exception as e:
        print(f"シミュレーション実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
