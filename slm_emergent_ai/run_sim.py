"""
メインスクリプト - SLM Emergent AI の実行エントリーポイント

複数の SLM エージェントを協調させ、Boids アルゴリズムによる創発知能を実証するシステムの
メインスクリプト。コマンドライン引数の処理、設定ファイルの読み込み、システムの初期化と
実行を行う。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional

import yaml
import numpy as np
import torch

from .agents.core import SLMAgent, LLM
from .memory.blackboard import BlackBoard
from .controller.meta import MetaController
from .eval.metrics import MetricsTracker
from .ui.dashboard import init_app
from .agents.rag import RAGAgent


# -----------------------------------------------------------------------------
# シミュレーション本体
# -----------------------------------------------------------------------------

async def run_simulation(config: Dict[str, Any]) -> None:  # noqa: C901  (長い関数を許容)
    """設定に従ってシミュレーションを実行する。"""

    print("Initializing SLM Emergent AI...")

    # ------------------------------------------------------------------
    # BlackBoard 初期化
    # ------------------------------------------------------------------
    bb_mode: str = config.get("memory", {}).get("mode", "local")
    redis_url: Optional[str] = config.get("memory", {}).get("redis_url")
    bb = BlackBoard(mode=bb_mode, redis_url=redis_url)

    # ------------------------------------------------------------------
    # モデル初期化
    # ------------------------------------------------------------------
    model_path: str = config.get("model", {}).get("path", "gemma3:1b")
    threads: int = config.get("runtime", {}).get("threads", 4)
    quantize: str = config.get("model", {}).get("quantize", "q4")
    n_ctx: int = config.get("model", {}).get("n_ctx", 512)

    print(
        f"Loading model: {model_path} with {threads} threads, "
        f"quantization: {quantize}, context length: {n_ctx}"
    )

    try:
        model = LLM(model_path, threads=threads, quantize=quantize, n_ctx=n_ctx)
        print("Model loaded successfully!")

        # 動作確認
        try:
            test_response = model.generate("テスト", max_tokens=10, temperature=0.5)
            print(f"[DEBUG] Model test successful: {test_response[:50]}...")
        except Exception as test_e:  # noqa: BLE001
            print(f"[WARNING] Model test failed: {test_e}")
            print("Proceeding with caution...")

    except FileNotFoundError as e:
        print(f"ERROR: Model file not found: {e}")
        print("Please ensure the model file exists at the specified path.")
        return
    except ImportError as e:
        print(f"ERROR: Required library not installed: {e}")
        print("Please install the required dependencies.")
        return
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Failed to initialize model: {e}")
        print(f"Exception type: {type(e).__name__}")
        print("Please check your model configuration and try again.")
        return

    # ------------------------------------------------------------------
    # エージェント初期化
    # ------------------------------------------------------------------
    agent_count: int = config.get("agent", {}).get("n", 3)
    agents: List[SLMAgent] = []

    # λ パラメータ
    lambda_params: Dict[str, float] = {
        "λ_a": config.get("lambda", {}).get("a", 0.3),
        "λ_c": config.get("lambda", {}).get("c", 0.3),
        "λ_s": config.get("lambda", {}).get("s", 0.1),
    }

    # BlackBoard へ λ パラメータ反映
    for key, value in lambda_params.items():
        await bb.set_param(key, value)

    await bb.set_param("base_speed", 10.0)

    print(f"Initializing {agent_count} agents...")

    roles: List[str] = config.get("agent", {}).get(
        "roles", ["質問者", "回答者", "批評者"]
    )
    agent_names: List[str] = [
        "アリス",
        "ボブ",
        "キャロル",
        "デイブ",
        "イブ",
        "フランク",
    ]

    for i in range(agent_count):
        role = roles[i % len(roles)]
        name = agent_names[i] if i < len(agent_names) else f"Agent{i + 1}"
        agent = SLMAgent(
            id=i + 1,
            role=role,
            model=model,
            tokenizer=model.tokenizer, # model is an instance of LLM, which should have a tokenizer attribute
            λ=lambda_params,
            name=name,
            blackboard=bb,  # Pass the blackboard instance
            boids_config=config.get("boids_enhancer", {}) # Pass boids specific config
        )
        agents.append(agent)
        print(f"[DEBUG] Agent {i + 1} ({name}) initialized with role: {role}")

    # ------------------------------------------------------------------
    # RAG エージェント（任意）
    # ------------------------------------------------------------------
    rag_enabled: bool = config.get("rag", {}).get("enabled", False)
    rag_agent: Optional[RAGAgent] = None
    if rag_enabled:
        backend_type = config.get("rag", {}).get("backend", "chroma")
        backend_config = config.get("rag", {}).get("config", {})
        print(f"Initializing RAG with {backend_type} backend...")
        rag_agent = RAGAgent(
            backend_type=backend_type, backend_config=backend_config, model_path=model_path
        )

    # ------------------------------------------------------------------
    # MetaController & MetricsTracker
    # ------------------------------------------------------------------
    controller = MetaController(
        bb=bb,
        target_H=config.get("controller", {}).get("target_H", 5.0),
        Kp=config.get("controller", {}).get("Kp", 0.1),
        Ki=config.get("controller", {}).get("Ki", 0.01),
        Kd=config.get("controller", {}).get("Kd", 0.05),
    )
    metrics_tracker = MetricsTracker()

    # ------------------------------------------------------------------
    # UI ダッシュボード
    # ------------------------------------------------------------------
    ui_conf = config.get("ui", {}) if isinstance(config.get("ui"), dict) else {}
    ui_enabled: bool = ui_conf.get("enabled", False)
    if ui_enabled:
        ui_port = ui_conf.get("port", 7860)
        app = init_app(blackboard=bb)

        import threading
        import uvicorn

        ui_thread = threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={"host": "0.0.0.0", "port": ui_port},
            daemon=True,
        )
        ui_thread.start()
        print(f"UI dashboard started on port {ui_port}")

    # ------------------------------------------------------------------
    # シミュレーション実行準備
    # ------------------------------------------------------------------
    print("Starting simulation...")
    simulation_logs: List[Dict[str, Any]] = []
    step_counter: int = 0

    eval_conf = config.get("evaluation", {})
    eval_enabled: bool = eval_conf.get("enabled", True)
    eval_freq: int = eval_conf.get("freq", 5)
    print(f"[DEBUG] Evaluation enabled: {eval_enabled}, frequency: {eval_freq}")

    initial_prompts: List[str] = config.get("prompts", {}).get(
        "initial", ["こんにちは、私はAIアシスタントです。"]
    )

    # Controller をバックグラウンド実行
    controller_task = asyncio.create_task(controller.run())

    # ------------------------------------------------------------------
    # メトリクス更新ループ
    # ------------------------------------------------------------------
    async def metrics_update_loop() -> None:
        nonlocal step_counter, simulation_logs, model, metrics_tracker, bb  # Ensure model and metrics_tracker are in scope
        total_tokens_generated_so_far = 0 # Initialize total tokens

        while True:
            await asyncio.sleep(2)  # 2 秒間隔 (configurable metric update interval)
            step_counter += 1

            # --- Data Collection ---
            # For VDI: Aggregate tokens from recent messages
            # For Speed: Aggregate all tokens generated so far
            all_messages_raw = await bb.pull_messages_raw(k=-1) # Get all messages for token count and VDI

            all_texts_for_vdi = []
            current_total_tokens = 0
            processed_texts_for_vdi = []

            if model.tokenizer: # Check if tokenizer is available
                for msg in all_messages_raw:
                    if isinstance(msg, dict) and "text" in msg:
                        text_content = msg.get("text", "")
                        # For VDI, use tokens from recent messages (e.g., last 20 as before, or a configurable number)
                        # For simplicity here, let's use text from all_messages_raw for VDI calculation for now
                        # but ideally, this should be from a more focused window.
                        processed_texts_for_vdi.append(text_content)
                        current_total_tokens += len(model.tokenizer.encode(text_content))

                # Aggregate tokens from recent messages for VDI
                # Using last N messages for VDI calculation, let's say last 50 messages or 1000 tokens.
                # The current calculate_vdi takes a list of tokens and a window_size for *that list*.
                # So we need to provide a list of tokens from recent messages.

                # Let's collect tokens from the last 20 messages for VDI as it was somewhat implied by original code.
                recent_messages_for_vdi_calc = all_messages_raw[-20:]
                vdi_texts = [msg.get("text", "") for msg in recent_messages_for_vdi_calc if isinstance(msg, dict)]
                vdi_tokens = []
                for text_content in vdi_texts:
                    vdi_tokens.extend(model.tokenizer.encode(text_content))

            else: # Fallback if tokenizer is not available (e.g. simple char/word count)
                for msg in all_messages_raw:
                    if isinstance(msg, dict) and "text" in msg:
                        text_content = msg.get("text", "")
                        processed_texts_for_vdi.append(text_content)
                        current_total_tokens += len(text_content.split()) # simple word count

                recent_messages_for_vdi_calc = all_messages_raw[-20:]
                vdi_texts = [msg.get("text", "") for msg in recent_messages_for_vdi_calc if isinstance(msg, dict)]
                # Simple tokenization for VDI if no tokenizer
                vdi_tokens = [word for text_content in vdi_texts for word in text_content.split()]


            total_tokens_generated_so_far = current_total_tokens

            # For FCR: Pass an empty list (placeholder)
            fact_check_results = []

            # For Entropy: No probability distribution source, so pass default or skip.
            # We will call metrics_tracker.update_entropy(0.0) directly.

            # --- Metric Calculation ---
            vdi_value = calculate_vdi(vdi_tokens) # calculate_vdi is imported from metrics.py
            fcr_value = calculate_fcr(fact_check_results) # calculate_fcr is imported

            # --- Tracker Updates ---
            metrics_tracker.update_vdi(vdi_value)
            metrics_tracker.update_fcr(fcr_value)
            metrics_tracker.update_entropy(0.0) # Default value for entropy
            metrics_tracker.update_token_count(total_tokens_generated_so_far) # This will trigger speed calculation

            # Logging (similar to before, but can now include new metrics from tracker)
            agent_responses = {
                msg.get("agent_id", 0): msg.get("text", "")
                for msg in all_messages_raw[-3:] # Use all_messages_raw to get latest
                if isinstance(msg, dict)
            }
            step_log = {
                "step": step_counter,
                "timestamp": time.strftime("%H:%M:%S"),
                "agent_responses": agent_responses,
                "total_messages": len(all_messages_raw),
                "calculated_vdi": vdi_value,
                "calculated_fcr": fcr_value,
                "total_tokens_for_speed": total_tokens_generated_so_far,
                "current_speed": metrics_tracker.current_metrics.get('speed', 0.0)
            }
            simulation_logs.append(step_log)

            if eval_enabled and step_counter % eval_freq == 0:
                # The old current_metrics dict is mostly replaced by direct calculations and tracker updates
                # We can log the metrics tracker's current state
                current_tracked_metrics = metrics_tracker.get_current_metrics()
                print(f"[METRICS] Step {step_counter}: {current_tracked_metrics}")
                # Lambda parameters are updated by the controller, not directly here.
                # If lambda params were part of current_metrics before, ensure controller still updates them in blackboard.
                # The metrics_tracker has update_lambda, but it should be called by controller if it changes them.


    metrics_task = asyncio.create_task(metrics_update_loop())

    # ------------------------------------------------------------------
    # エージェント実行 (staggered start)
    # ------------------------------------------------------------------
    print("Starting agents with staggered execution...")
    agent_tasks: List[asyncio.Task] = []

    for i, agent in enumerate(agents):
        prompt = initial_prompts[i % len(initial_prompts)]
        delay = i * 3.0  # 3 秒ずつずらす

        async def delayed_start(a: SLMAgent, p: str, d: float) -> None:
            await asyncio.sleep(d)
            print(f"[DEBUG] Starting agent {a.id} ({a.name}/{a.role}) after {d}s delay")
            await a.run_conversation(p, bb, max_turns=sys.maxsize)

        agent_tasks.append(asyncio.create_task(delayed_start(agent, prompt, delay)))

    # ------------------------------------------------------------------
    # 全タスク並列実行
    # ------------------------------------------------------------------
    try:
        print("[INFO] All agents started with staggered timing for stability")
        await asyncio.gather(*agent_tasks, metrics_task)
    except KeyboardInterrupt:
        print("\nCtrl+C が押されました。シミュレーションを安全に停止します...")
    except Exception as e:  # noqa: BLE001
        print(f"\n[ERROR] シミュレーション実行中にエラー: {e}")
    finally:
        # すべてのタスクをキャンセル
        for task in [controller_task, metrics_task, *agent_tasks]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        print("クリーンアップ完了。")
        print(
            f"[SUMMARY] Total simulation steps: {step_counter}, "
            f"Total logs: {len(simulation_logs)}"
        )

    print("Simulation completed!")


# -----------------------------------------------------------------------------
# エントリーポイント
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401 - imperative mood
    """CLI 引数を解釈しシミュレーションを起動する。"""

    parser = argparse.ArgumentParser(description="SLM Emergent AI")
    parser.add_argument("-c", "--config", default="configs/base.yml", help="設定ファイルのパス")
    parser.add_argument("-m", "--multi", action="store_true", help="複数設定を並列実行")

    # オーバーライド用フラットパラメータ
    parser.add_argument("--agent.n", type=int, help="エージェント数")
    parser.add_argument("--model.path", type=str, help="モデルパス")
    parser.add_argument("--runtime.threads", type=int, help="スレッド数")
    parser.add_argument("--ui.enabled", type=bool, help="UI の有効化")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 設定ファイル読み込み
    # ------------------------------------------------------------------
    config_path = (
        args.config
        if os.path.isabs(args.config)
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    )

    if not os.path.exists(config_path):
        config_path = os.path.abspath(args.config)  # CWD からの相対パス再チャック
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # CLI からのオーバーライド反映
    # ------------------------------------------------------------------
    for arg_name, arg_value in vars(args).items():
        if arg_value is None or arg_name in {"config", "multi"}:
            continue
        keys = arg_name.split(".")
        curr = config
        for key in keys[:-1]:
            curr = curr.setdefault(key, {})
        curr[keys[-1]] = arg_value

    # ------------------------------------------------------------------
    # シミュレーション実行
    # ------------------------------------------------------------------
    asyncio.run(run_simulation(config))


if __name__ == "__main__":
    main()
