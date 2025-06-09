"""
メインスクリプト - SLM Emergent AI の実行エントリーポイント
Main script - Entry point for SLM Emergent AI execution.

複数の SLM エージェントを協調させ、Boids アルゴリズムによる創発知能を実証するシステムの
メインスクリプト。コマンドライン引数の処理、設定ファイルの読み込み、システムの初期化と
実行を行う。
This main script handles command-line arguments, loads configuration,
and initializes and runs the system where multiple SLM agents cooperate
to demonstrate emergent intelligence, previously inspired by Boids algorithms
and now using a logits-based modification approach.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional

import yaml
# numpy and torch are not directly used in this file after recent refactorings,
# but are kept as they are fundamental to underlying libraries.
import numpy as np
import torch

from .agents.core import SLMAgent, LLM
from .memory.blackboard import BlackBoard
from .controller.meta import MetaController
from .eval.metrics import MetricsTracker, calculate_vdi, calculate_fcr
from .ui.dashboard import init_app
from .agents.rag import RAGAgent


# -----------------------------------------------------------------------------
# Simulation Core
# -----------------------------------------------------------------------------

async def run_simulation(config: Dict[str, Any]) -> None:  # noqa: C901 (allow long function)
    """
    Initializes and runs the SLM Emergent AI simulation based on the provided configuration.

    Args:
        config (Dict[str, Any]): A dictionary containing the simulation configuration,
                                 typically loaded from a YAML file.
    """

    print("Initializing SLM Emergent AI...")

    # --- Blackboard Initialization ---
    bb_mode: str = config.get("memory", {}).get("mode", "local")
    redis_url: Optional[str] = config.get("memory", {}).get("redis_url")
    bb = BlackBoard(mode=bb_mode, redis_url=redis_url)

    # --- Model (LLM Wrapper) Initialization ---
    model_config = config.get("model", {})

    model_path: str = model_config.get("path", "google/gemma-2-2b-it")
    threads: int = config.get("runtime", {}).get("threads", 4)
    quantize: str = model_config.get("quantize", "q4")
    n_ctx: int = model_config.get("n_ctx", 512)

    boids_logits_params = model_config.get("boids_logits_params", {})

    initial_prompts_config: List[str] = config.get("prompts", {}).get(
        "initial", ["Default initial prompt for collaborative discussion."]
    )
    hardcoded_default_cohesion_text = "Focus on collaborative problem solving, shared understanding, and innovative ideas."
    default_cohesion_from_prompts = initial_prompts_config[0] if initial_prompts_config else hardcoded_default_cohesion_text

    llm_cohesion_prompt_text = boids_logits_params.get("cohesion_prompt_text")
    if not llm_cohesion_prompt_text:
        llm_cohesion_prompt_text = default_cohesion_from_prompts

    print(
        f"Loading model: {model_path} (threads: {threads}, quant: {quantize}, context: {n_ctx})"
    )
    # print(f"Boids Logits Params being used for LLM: {boids_logits_params}") # Reduced noise
    # print(f"Cohesion prompt text for LLM's Boids processor: '{llm_cohesion_prompt_text}'")


    try:
        model = LLM(
            model_path=model_path,
            threads=threads,
            quantize=quantize,
            n_ctx=n_ctx,
            boids_enabled=boids_logits_params.get("enabled", True),
            w_align=boids_logits_params.get("w_align", 0.1),
            w_sep=boids_logits_params.get("w_sep", 0.1),
            w_cohesion=boids_logits_params.get("w_cohesion", 0.1),
            n_align_tokens=boids_logits_params.get("n_align_tokens", 10),
            m_sep_tokens=boids_logits_params.get("m_sep_tokens", 10),
            theta_sep=boids_logits_params.get("theta_sep", 0.8),
            cohesion_prompt_text=llm_cohesion_prompt_text
        )
        print("Model (LLM wrapper) loaded successfully with Boids config (if enabled).")

        try:
            test_response = model.generate("Test prompt", max_tokens=10, temperature=0.5)
            print(f"[DEBUG] Model test successful: {test_response[:50]}...")
        except Exception as test_e:
            print(f"[WARNING] Model test generation failed: {test_e}")
            print("Proceeding with caution...")

    except FileNotFoundError as e:
        print(f"ERROR: Model file not found: {e}. Please ensure the model file exists at the specified path.")
        return
    except ImportError as e:
        print(f"ERROR: Required library not installed: {e}. Please install the required dependencies.")
        return
    except Exception as e:
        print(f"ERROR: Failed to initialize model: {e}")
        print(f"Exception type: {type(e).__name__}")
        print("Please check your model configuration and try again.")
        return

    # --- Agent Initialization ---
    agent_count: int = config.get("agent", {}).get("n", 3)
    agents: List[SLMAgent] = []

    print(f"Initializing {agent_count} agents...")

    roles_list_config: List[str] = config.get("agent", {}).get("roles", ["Agent"])

    for i in range(agent_count):
        role = roles_list_config[i % len(roles_list_config)]
        name = f"Agent_{i+1}"

        agent = SLMAgent(
            id=i + 1,
            role=role,
            model=model,
            tokenizer=model.tokenizer,
            name=name,
        )
        agents.append(agent)
        print(f"[DEBUG] Agent {i + 1} ({name}) initialized with role: {role}.")

    # --- RAG Agent Initialization (Optional) ---
    rag_model_path = model_config.get("path", "google/gemma-2-2b-it")
    rag_enabled: bool = config.get("rag", {}).get("enabled", False)
    if rag_enabled:
        backend_type = config.get("rag", {}).get("backend", "chroma")
        rag_backend_config = config.get("rag", {}).get("config", {})
        print(f"Initializing RAG with {backend_type} backend...")
        rag_agent = RAGAgent(
            backend_type=backend_type, backend_config=rag_backend_config, model_path=rag_model_path
        )

    # --- MetaController & MetricsTracker Initialization ---
    controller = MetaController(
        bb=bb,
        target_H=config.get("controller", {}).get("target_H", 5.0),
        Kp=config.get("controller", {}).get("Kp", 0.1),
        Ki=config.get("controller", {}).get("Ki", 0.01),
        Kd=config.get("controller", {}).get("Kd", 0.05),
    )
    metrics_tracker = MetricsTracker()

    # --- UI Dashboard Initialization (Optional) ---
    ui_conf = config.get("ui", {}) if isinstance(config.get("ui"), dict) else {}
    ui_enabled: bool = ui_conf.get("enabled", False)
    if ui_enabled:
        ui_port = ui_conf.get("port", 7860)
        app_ui = init_app(blackboard=bb)

        import threading
        import uvicorn

        ui_thread = threading.Thread(
            target=uvicorn.run,
            args=(app_ui,),
            kwargs={"host": "0.0.0.0", "port": ui_port},
            daemon=True,
        )
        ui_thread.start()
        print(f"UI dashboard started on port {ui_port}")

    # --- Simulation Execution Preparation ---
    print("Starting simulation...")
    simulation_logs: List[Dict[str, Any]] = []
    step_counter: int = 0

    eval_conf = config.get("evaluation", {})
    eval_enabled: bool = eval_conf.get("enabled", True)
    eval_freq: int = eval_conf.get("freq", 5)
    print(f"Evaluation enabled: {eval_enabled}, logging frequency: {eval_freq} steps")

    initial_prompts_for_conversation = initial_prompts_config

    controller_task = asyncio.create_task(controller.run())

    # --- Metrics Update Loop ---
    async def metrics_update_loop() -> None:
        nonlocal step_counter, simulation_logs, model, metrics_tracker, bb

        while True:
            await asyncio.sleep(2)
            step_counter += 1

            all_messages_raw = await bb.pull_messages_raw(k=-1)

            current_total_tokens = 0
            vdi_tokens_list = []

            active_tokenizer = model.tokenizer

            for msg_item in all_messages_raw:
                if isinstance(msg_item, dict) and "text" in msg_item:
                    text_content = msg_item.get("text", "")
                    if active_tokenizer:
                        current_total_tokens += len(active_tokenizer.encode(text_content))
                    else:
                        current_total_tokens += len(text_content.split())

            recent_messages_for_vdi_calc = all_messages_raw[-20:]
            vdi_texts = [msg_item.get("text", "") for msg_item in recent_messages_for_vdi_calc if isinstance(msg_item, dict)]
            for text_item in vdi_texts:
                if active_tokenizer:
                    vdi_tokens_list.extend(active_tokenizer.encode(text_item))
                else:
                    vdi_tokens_list.extend(text_item.split())

            # Updated placeholder comment
            fact_check_results = [] # Placeholder for FCR data; FCR will be 1.0 as no fact-checking mechanism is currently implemented.

            # Metric Calculation (calculate_vdi, calculate_fcr are imported from top)
            print(f"[METRICS_DEBUG] VDI Tokens (len): {len(vdi_tokens_list)}, Sample: {vdi_tokens_list[:10] if vdi_tokens_list else '[]'}...")
            vdi_value = calculate_vdi(vdi_tokens_list)
            fcr_value = calculate_fcr(fact_check_results)
            print(f"[METRICS_DEBUG] Calculated VDI: {vdi_value:.4f}, Calculated FCR: {fcr_value:.4f}")

            # Tracker Updates
            metrics_tracker.update_vdi(vdi_value)
            metrics_tracker.update_fcr(fcr_value)
            # Updated placeholder comment
            metrics_tracker.update_entropy(0.0) # Placeholder for entropy; actual entropy from model probabilities is not currently implemented.
            print(f"[METRICS_DEBUG] Total tokens for speed calc (current_total_tokens): {current_total_tokens}")
            metrics_tracker.update_token_count(current_total_tokens)

            # Logging step_log
            agent_responses_for_log = {
                msg_item.get("agent_id", 0): msg_item.get("text", "")
                for msg_item in all_messages_raw[-3:]
                if isinstance(msg_item, dict)
            }
            # This is the dictionary that gets logged to console if eval_enabled
            current_metrics_to_log = metrics_tracker.get_current_metrics()
            # Augment with step and other info for simulation_logs list
            step_log_entry = {
                "step": step_counter, "timestamp": time.strftime("%H:%M:%S"),
                "agent_responses": agent_responses_for_log, "total_messages": len(all_messages_raw),
                "vdi": current_metrics_to_log.get('vdi', 0.0), # Get from tracker
                "fcr": current_metrics_to_log.get('fcr', 0.0), # Get from tracker
                "entropy": current_metrics_to_log.get('entropy', 0.0), # Get from tracker
                "speed": current_metrics_to_log.get('speed', 0.0), # Get from tracker
                "total_tokens_for_speed_calc": current_total_tokens,
            }
            simulation_logs.append(step_log_entry)

            if eval_enabled and step_counter % eval_freq == 0:
                # Log the dictionary that is being constructed for simulation_logs for console output
                # but ensure it includes all relevant current metrics from the tracker.
                # The `current_metrics_to_log` now directly comes from tracker.
                print(f"[METRICS_DEBUG] Logged current_metrics_to_log for console: {current_metrics_to_log}")
                print(f"[METRICS] Step {step_counter}: {current_metrics_to_log}") # Original log line

    metrics_task = asyncio.create_task(metrics_update_loop())

    # --- Agent Execution (staggered start) ---
    print("Starting agents with staggered execution...")
    agent_tasks: List[asyncio.Task] = []

    for i, agent_instance in enumerate(agents):
        prompt_for_agent = initial_prompts_for_conversation[i % len(initial_prompts_for_conversation)]
        delay = i * 3.0

        async def delayed_start(agent_to_run: SLMAgent, p_task: str, d_delay: float) -> None:
            await asyncio.sleep(d_delay)
            print(f"[DEBUG] Starting agent {agent_to_run.id} ({agent_to_run.name}/{agent_to_run.role}) after {d_delay}s delay")
            await agent_to_run.run_conversation(p_task, bb, max_turns=sys.maxsize)

        agent_tasks.append(asyncio.create_task(delayed_start(agent_instance, prompt_for_agent, delay)))

    # --- Run All Tasks ---
    try:
        print("[INFO] All agents started with staggered timing for stability. Simulation running...")
        await asyncio.gather(*agent_tasks, metrics_task)
    except KeyboardInterrupt:
        print("\nCtrl+C pressed. Safely stopping simulation...")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during simulation execution: {e}")
    finally:
        print("Cancelling outstanding tasks...")
        for task_item in [controller_task, metrics_task, *agent_tasks]:
            if task_item and not task_item.done():
                task_item.cancel()
                try:
                    await task_item
                except asyncio.CancelledError:
                    task_name = task_item.get_name() if hasattr(task_item, 'get_name') else 'Unnamed task'
                    print(f"Task {task_name} was cancelled.")
                except Exception as e_cancel:
                    print(f"Error cancelling task: {e_cancel}")

        print("Cleanup complete.")
        print(f"[SUMMARY] Total simulation steps: {step_counter}, Total logs recorded: {len(simulation_logs)}")

    print("Simulation completed!")


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

def main() -> None:
    """
    Command-Line Interface (CLI) entry point.
    Parses arguments, loads configuration, and launches the simulation.
    """
    parser = argparse.ArgumentParser(description="SLM Emergent AI Simulation Framework")
    parser.add_argument(
        "-c", "--config",
        default=os.path.join(os.path.dirname(__file__), "configs", "base.yml"),
        help="Path to the simulation configuration YAML file."
    )
    parser.add_argument("-m", "--multi", action="store_true", help="Run multiple configurations (not fully implemented).")

    parser.add_argument("--agent.n", type=int, help="Override number of agents.")
    parser.add_argument("--model.path", type=str, help="Override model path.")
    parser.add_argument("--runtime.threads", type=int, help="Override number of runtime threads.")
    parser.add_argument("--ui.enabled", type=str, help="Override UI enabled state (true/false).")

    args = parser.parse_args()

    config_path_to_load = args.config
    if not os.path.isabs(config_path_to_load):
        if not os.path.exists(config_path_to_load) and __file__ and os.path.dirname(__file__):
             script_dir = os.path.dirname(os.path.abspath(__file__))
             config_path_to_load = os.path.join(script_dir, args.config)

    if not os.path.exists(config_path_to_load):
        config_path_to_load = os.path.abspath(args.config)

    if not os.path.exists(config_path_to_load):
        print(f"Configuration file not found: {args.config} (checked: {config_path_to_load})")
        sys.exit(1)

    with open(config_path_to_load, "r", encoding="utf-8") as f:
        config_data: Dict[str, Any] = yaml.safe_load(f)

    for arg_name_cli, arg_value_cli in vars(args).items():
        if arg_value_cli is None or arg_name_cli in {"config", "multi"}:
            continue

        keys_cli = arg_name_cli.split('.')
        temp_config_ptr = config_data
        for key_part_cli in keys_cli[:-1]:
            if key_part_cli not in temp_config_ptr or not isinstance(temp_config_ptr[key_part_cli], dict):
                temp_config_ptr[key_part_cli] = {}
            temp_config_ptr = temp_config_ptr[key_part_cli]

        final_key_cli = keys_cli[-1]

        # Check existing type in config to guide conversion if possible
        current_val_in_config = temp_config_ptr.get(final_key_cli)
        target_type = type(current_val_in_config) if current_val_in_config is not None else None

        try:
            if target_type == bool:
                if str(arg_value_cli).lower() in ['true', 't', 'yes', 'y', '1']: temp_config_ptr[final_key_cli] = True
                elif str(arg_value_cli).lower() in ['false', 'f', 'no', 'n', '0']: temp_config_ptr[final_key_cli] = False
                else: print(f"Warning: Could not parse boolean for {arg_name_cli}: {arg_value_cli}")
            elif target_type == float: temp_config_ptr[final_key_cli] = float(arg_value_cli)
            elif target_type == int: temp_config_ptr[final_key_cli] = int(arg_value_cli)
            else: temp_config_ptr[final_key_cli] = arg_value_cli # Handles str or if target_type is None (new key)
        except ValueError:
            print(f"Warning: Could not parse argument {arg_name_cli} with value '{arg_value_cli}' to type {target_type}.")


    asyncio.run(run_simulation(config_data))


if __name__ == "__main__":
    main()

```
