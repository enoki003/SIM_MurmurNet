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
import signal
from typing import Any, Dict, List, Optional, Callable # Added Callable

import yaml
# numpy and torch are not directly used in this file after recent refactorings,
# but are kept as they are fundamental to underlying libraries.
import numpy as np
import torch

# Add current directory to Python path for module discovery
if __name__ == "__main__":
    # Add the parent directory to sys.path to enable imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# Try absolute import first, fall back to relative import
try:
    from slm_emergent_ai.agents.core import SLMAgent, LLM
    from slm_emergent_ai.memory.blackboard import BlackBoard
    from slm_emergent_ai.controller.meta import MetaController
    from slm_emergent_ai.eval.metrics import MetricsTracker, calculate_vdi, calculate_fcr
    from slm_emergent_ai.ui.dashboard import init_app
    from slm_emergent_ai.agents.rag import RAGAgent
    from slm_emergent_ai.summarizer.conversation_summarizer import ConversationSummarizer
    from slm_emergent_ai.agents.history_manager import AgentHistoryManager
except ImportError:
    # Fallback to relative imports
    from .agents.core import SLMAgent, LLM
    from .summarizer.conversation_summarizer import ConversationSummarizer
    from .agents.history_manager import AgentHistoryManager
    from .memory.blackboard import BlackBoard
    from .controller.meta import MetaController
    from .eval.metrics import MetricsTracker, calculate_vdi, calculate_fcr
    from .ui.dashboard import init_app
    from .agents.rag import RAGAgent
    # Ensure ConversationSummarizer and AgentHistoryManager are part of this fallback too if not already
    # For brevity, assuming they are covered by the addition above or would be added similarly.


# グローバル変数でシャットダウン状態を管理
shutdown_requested = False
current_agents = []

def signal_handler(signum, frame):
    """シグナルハンドラー（Ctrl+C対応）"""
    global shutdown_requested, current_agents
    print(f"\n[INFO] Shutdown signal received (signal: {signum}). Initiating graceful shutdown...")
    shutdown_requested = True
    
    # エージェントにシャットダウンを通知
    for agent in current_agents:
        if hasattr(agent, 'shutdown'):
            agent.shutdown()

# -----------------------------------------------------------------------------
# Helper Functions for Initialization
# -----------------------------------------------------------------------------

def _setup_signal_handlers():
    """Sets up global signal handlers."""
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    print("Signal handlers set up.")

def _setup_blackboard(config: Dict[str, Any]) -> BlackBoard:
    """Initializes and returns the BlackBoard instance."""
    bb_mode: str = config.get("memory", {}).get("mode", "local")
    redis_url: Optional[str] = config.get("memory", {}).get("redis_url")
    bb = BlackBoard(mode=bb_mode, redis_url=redis_url)
    print(f"BlackBoard initialized (mode: {bb_mode}).")
    return bb

def _setup_llm(config: Dict[str, Any]) -> Optional[LLM]:
    """Initializes and returns the LLM instance."""
    model_config = config.get("model", {})
    model_path: str = model_config.get("path", "google/gemma-2-2b-it")
    threads: int = config.get("runtime", {}).get("threads", 4)
    quantize: str = model_config.get("quantize", "q4")
    n_ctx: int = model_config.get("n_ctx", 512)

    print(
        f"Loading model: {model_path} (threads: {threads}, quant: {quantize}, context: {n_ctx})"
    )
    try:
        model = LLM(
            model_path=model_path,
            threads=threads,
            quantize=quantize,
            n_ctx=n_ctx
        )
        print("Model (LLM wrapper) loaded successfully.")
        try:
            test_response = model.generate("Test prompt", max_tokens=10, temperature=0.5)
            print(f"[DEBUG] Model test successful: {test_response[:50]}...")
        except Exception as test_e:
            print(f"[WARNING] Model test generation failed: {test_e}")
            print("Proceeding with caution...")
        return model
    except FileNotFoundError as e:
        print(f"ERROR: Model file not found: {e}. Please ensure the model file exists at the specified path.")
        return None
    except ImportError as e:
        print(f"ERROR: Required library not installed: {e}. Please install the required dependencies.")
        return None
    except Exception as e:
        print(f"ERROR: Failed to initialize model: {e}")
        print(f"Exception type: {type(e).__name__}")
        print("Please check your model configuration and try again.")
        return None

def _setup_summarizer_and_history_manager(
    config: Dict[str, Any], blackboard: BlackBoard, llm_model: LLM
) -> tuple[Optional[ConversationSummarizer], AgentHistoryManager]:
    """Initializes and returns ConversationSummarizer and AgentHistoryManager instances."""
    # ConversationSummarizer Initialization
    summarizer_instance: Optional[ConversationSummarizer] = None
    try:
        # TODO: Consider making ConversationSummarizer parameters configurable
        summarizer_instance = ConversationSummarizer() # Uses default params
        print("ConversationSummarizer initialized with default parameters.")
    except Exception as e:
        print(f"[WARNING] Failed to initialize ConversationSummarizer: {e}. Summarization will be disabled.")
        # summarizer_instance remains None

    # AgentHistoryManager Initialization
    # Note: llm_model is passed for potential future use by summarizer, not currently used by AgentHistoryManager directly.
    history_manager = AgentHistoryManager(blackboard=blackboard, summarizer=summarizer_instance)
    print("AgentHistoryManager initialized.")
    return summarizer_instance, history_manager

def _setup_agents(
    config: Dict[str, Any], llm_model: LLM, history_manager_instance: AgentHistoryManager
) -> List[SLMAgent]:
    """Initializes and returns a list of SLMAgents."""
    global current_agents # For signal_handler
    agent_config = config.get("agent", {})
    agent_count: int = agent_config.get("n", 3)
    agents_list: List[SLMAgent] = []

    print(f"Initializing {agent_count} agents...")
    roles_list_config: List[str] = agent_config.get("roles", ["Agent"])

    for i in range(agent_count):
        role = roles_list_config[i % len(roles_list_config)]
        name = f"Agent_{i+1}"
        agent = SLMAgent(
            id=i + 1,
            role=role,
            model=llm_model,
            name=name,
            history_manager=history_manager_instance
        )
        agents_list.append(agent)
        print(f"[DEBUG] Agent {i + 1} ({name}) initialized with role: {role}.")

    current_agents = agents_list # Update global list
    return agents_list

def _setup_rag_agent(config: Dict[str, Any], default_model_path: str) -> Optional[RAGAgent]:
    """Initializes and returns the RAGAgent instance if enabled."""
    rag_config = config.get("rag", {})
    rag_enabled: bool = rag_config.get("enabled", False)

    if rag_enabled:
        model_config = config.get("model", {}) # RAG might use a different model path or same as main
        rag_model_path = rag_config.get("model_path", default_model_path) # Use specific RAG model or default
        backend_type = rag_config.get("backend", "chroma")
        rag_backend_config = rag_config.get("config", {})
        print(f"Initializing RAG with {backend_type} backend (model: {rag_model_path})...")
        try:
            rag_agent_instance = RAGAgent(
                backend_type=backend_type, backend_config=rag_backend_config, model_path=rag_model_path
            )
            print("RAGAgent initialized successfully.")
            return rag_agent_instance
        except Exception as e:
            print(f"[WARNING] Failed to initialize RAGAgent: {e}. RAG features will be unavailable.")
            return None
    return None

def _setup_controller(config: Dict[str, Any], blackboard: BlackBoard) -> MetaController:
    """Initializes and returns the MetaController instance."""
    controller_config = config.get("controller", {})
    controller = MetaController(
        bb=blackboard,
        target_H=controller_config.get("target_H", 5.0),
        Kp=controller_config.get("Kp", 0.1),
        Ki=controller_config.get("Ki", 0.01),
        Kd=controller_config.get("Kd", 0.05),
    )
    print("MetaController initialized.")
    return controller

def _setup_metrics_tracker() -> MetricsTracker:
    """Initializes and returns the MetricsTracker instance."""
    metrics_tracker = MetricsTracker()
    print("MetricsTracker initialized.")
    return metrics_tracker

def _setup_ui(config: Dict[str, Any], blackboard: BlackBoard) -> Optional[threading.Thread]:
    """Initializes and starts the UI dashboard if enabled, returning the UI thread."""
    ui_config = config.get("ui", {})
    ui_enabled: bool = ui_config.get("enabled", False)

    if ui_enabled:
        ui_port = ui_config.get("port", 7860)
        app_ui = init_app(blackboard=blackboard)

        import threading # Keep import local to where it's used
        import uvicorn

        ui_thread = threading.Thread(
            target=uvicorn.run,
            args=(app_ui,),
            kwargs={"host": "0.0.0.0", "port": ui_port},
            daemon=True,
        )
        ui_thread.start()
        print(f"UI dashboard started on port {ui_port}.")
        return ui_thread
    print("UI dashboard is disabled.")
    return None

# -----------------------------------------------------------------------------
# Metrics Update Loop (Top-Level)
# -----------------------------------------------------------------------------

async def metrics_update_loop(
    shutdown_flag_check: Callable[[], bool],
    bb: BlackBoard,
    model: LLM,
    metrics_tracker: MetricsTracker,
    simulation_logs: List[Dict[str, Any]],
    eval_config: Dict[str, Any]
) -> int:
    """
    Periodically calculates and logs simulation metrics.
    Returns the total number of steps executed.
    """
    step_counter = 0
    eval_enabled = eval_config.get('enabled', True)
    eval_freq = eval_config.get('freq', 5)

    while not shutdown_flag_check():
        try:
            await asyncio.sleep(2) # Interval for metrics update
            step_counter += 1

            all_messages_raw = await bb.pull_messages_raw(k=-1)
            current_total_tokens = 0
            vdi_tokens_list = []
            active_tokenizer = model.tokenizer # Assuming model has tokenizer

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

            vdi_value = calculate_vdi(vdi_tokens_list)

            if len(all_messages_raw) > 1:
                unique_agent_responses = set()
                total_agent_responses = 0
                for msg in all_messages_raw[-10:]:
                    if isinstance(msg, dict) and msg.get("text"):
                        text = msg.get("text", "").strip()
                        if len(text) > 10:
                            unique_agent_responses.add(text[:50])
                            total_agent_responses += 1
                fcr_value = len(unique_agent_responses) / max(total_agent_responses, 1) if total_agent_responses > 0 else 1.0
            else:
                fcr_value = 1.0

            if len(all_messages_raw) > 0:
                agent_message_counts = {}
                for msg in all_messages_raw[-20:]:
                    if isinstance(msg, dict):
                        agent_id = msg.get("agent_id", 0)
                        agent_message_counts[agent_id] = agent_message_counts.get(agent_id, 0) + 1
                
                if agent_message_counts:
                    total_messages_count = sum(agent_message_counts.values())
                    probs = [count / total_messages_count for count in agent_message_counts.values()]
                    entropy_value = -sum(p * np.log2(p) for p in probs if p > 0) # np is still used here
                else:
                    entropy_value = 0.0
            else:
                entropy_value = 0.0

            metrics_tracker.update_vdi(vdi_value)
            metrics_tracker.update_fcr(fcr_value)
            metrics_tracker.update_entropy(entropy_value)
            metrics_tracker.update_token_count(current_total_tokens)

            current_metrics = metrics_tracker.get_current_metrics()
            await bb.set_param("vdi", current_metrics['vdi'])
            await bb.set_param("fcr", current_metrics['fcr'])
            await bb.set_param("entropy", current_metrics['entropy'])
            await bb.set_param("speed", current_metrics['speed'])

            agent_responses_for_log = {
                msg_item.get("agent_id", 0): msg_item.get("text", "")
                for msg_item in all_messages_raw[-3:]
                if isinstance(msg_item, dict)
            }
            step_log_entry = {
                "step": step_counter, "timestamp": time.strftime("%H:%M:%S"),
                "agent_responses": agent_responses_for_log, "total_messages": len(all_messages_raw),
                "vdi": current_metrics.get('vdi', 0.0),
                "fcr": current_metrics.get('fcr', 0.0),
                "entropy": current_metrics.get('entropy', 0.0),
                "speed": current_metrics.get('speed', 0.0),
                "total_tokens_for_speed_calc": current_total_tokens,
            }
            simulation_logs.append(step_log_entry)

            if eval_enabled and step_counter % eval_freq == 0:
                print(f"[METRICS_DEBUG] VDI Tokens (len): {len(vdi_tokens_list)}, Sample: {vdi_tokens_list[:10] if vdi_tokens_list else '[]'}...")
                print(f"[METRICS_DEBUG] Calculated VDI: {vdi_value:.4f}, Calculated FCR: {fcr_value:.4f}, Calculated Entropy: {entropy_value:.4f}")
                print(f"[METRICS_DEBUG] Total tokens for speed calc (current_total_tokens): {current_total_tokens}")
                print(f"[METRICS_DEBUG] Saved metrics to BlackBoard: VDI={current_metrics['vdi']:.4f}, FCR={current_metrics['fcr']:.4f}, Speed={current_metrics['speed']:.4f}")
                print(f"[METRICS_DEBUG] Logged current_metrics_to_log for console: {current_metrics}")
                print(f"[METRICS] Step {step_counter}: {current_metrics}")
        except asyncio.CancelledError:
            print("[INFO] Metrics update loop cancelled.")
            break
        except Exception as e:
            print(f"[ERROR] Error in metrics update loop: {e}")
            if shutdown_flag_check():
                break
    return step_counter

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
    global shutdown_requested, current_agents

    print("Initializing SLM Emergent AI...")

    _setup_signal_handlers()
    bb = _setup_blackboard(config)
    model = _setup_llm(config)

    if model is None:
        print("LLM initialization failed. Exiting simulation.")
        return

    # Retrieve initial_prompts_config after LLM setup.
    initial_prompts_config: List[str] = config.get("prompts", {}).get(
        "initial", ["Default initial prompt for collaborative discussion."]
    )

    summarizer_instance, history_manager = _setup_summarizer_and_history_manager(config, bb, model)
    agents = _setup_agents(config, model, history_manager)

    # Default model path for RAG can be the main model path if not specified in RAG config
    main_model_path = config.get("model", {}).get("path", "google/gemma-2-2b-it")
    rag_agent = _setup_rag_agent(config, main_model_path) # rag_agent might be None

    controller = _setup_controller(config, bb)
    metrics_tracker = _setup_metrics_tracker()
    ui_thread = _setup_ui(config, bb) # ui_thread might be None

    # --- Simulation Execution Preparation ---
    print("Starting simulation...")
    simulation_logs: List[Dict[str, Any]] = [] # Initialize locally

    eval_conf = config.get("evaluation", {})
    # eval_enabled and eval_freq are now handled inside metrics_update_loop via eval_config

    initial_prompts_for_conversation = initial_prompts_config
    controller_task = asyncio.create_task(controller.run())

    metrics_task = asyncio.create_task(
        metrics_update_loop(
            shutdown_flag_check=lambda: shutdown_requested, # Pass a callable for the global flag
            bb=bb,
            model=model,
            metrics_tracker=metrics_tracker,
            simulation_logs=simulation_logs, # Pass the local list
            eval_config=eval_conf
        )
    )

    # --- Agent Execution (staggered start) ---
    print("Starting agents with staggered execution...")
    agent_tasks: List[asyncio.Task] = []

    for i, agent_instance in enumerate(agents):
        prompt_for_agent = initial_prompts_for_conversation[i % len(initial_prompts_for_conversation)]
        delay = i * 3.0

        async def delayed_start(agent_to_run: SLMAgent, p_task: str, d_delay: float) -> None:
            await asyncio.sleep(d_delay)
            if not shutdown_requested:
                print(f"[DEBUG] Starting agent {agent_to_run.id} ({agent_to_run.name}/{agent_to_run.role}) after {d_delay}s delay")
                await agent_to_run.run_conversation(p_task, bb, max_turns=sys.maxsize)

        agent_tasks.append(asyncio.create_task(delayed_start(agent_instance, prompt_for_agent, delay)))

    # --- Run All Tasks ---
    try:
        print("[INFO] All agents started with staggered timing for stability. Simulation running...")
        print("[INFO] Press Ctrl+C to safely shutdown the simulation.")
        await asyncio.gather(*agent_tasks, metrics_task, return_exceptions=True)
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt caught. Initiating graceful shutdown...")
        shutdown_requested = True
    except Exception as e:
        print(f"\n[ERROR] An error occurred during simulation execution: {e}")
        shutdown_requested = True
    finally:
        print("\n[INFO] Shutting down simulation...")
        
        # エージェントにシャットダウンを通知
        for agent in agents:
            if hasattr(agent, 'shutdown'):
                agent.shutdown()
        
        print("Cancelling outstanding tasks...")
        tasks_to_cancel = [controller_task, metrics_task, *agent_tasks]
        for task_item in tasks_to_cancel:
            if task_item and not task_item.done():
                task_item.cancel()
        
        # タスクの完了を待機
        gathered_tasks = asyncio.gather(*[t for t in tasks_to_cancel if t], return_exceptions=True)
        try:
            await asyncio.wait_for(gathered_tasks, timeout=10.0) # Increased timeout slightly
        except asyncio.TimeoutError:
            print("[WARNING] Some tasks did not complete within timeout during shutdown.")
        except Exception as e:
            print(f"[ERROR] Exception during task gathering on shutdown: {e}")


        final_step_count = 0
        if metrics_task and metrics_task.done() and not metrics_task.cancelled():
            try:
                final_step_count = metrics_task.result()
            except Exception as e:
                print(f"[INFO] Could not retrieve step count from metrics_task: {e}")

        print("Cleanup complete.")
        print(f"[SUMMARY] Total simulation steps: {final_step_count}, Total logs recorded: {len(simulation_logs)}")

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

repr("")
