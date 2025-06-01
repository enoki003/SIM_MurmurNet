#!/usr/bin/env python3
"""
10エージェントベンチマーク - パフォーマンス測定

10エージェントを使用した場合のパフォーマンスを測定するベンチマークスクリプト。
CPU使用率、メモリ使用量、生成速度などを測定する。
"""

import os
import sys
import time
import asyncio
import argparse
import psutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from slm_emergent_ai.agents.core import SLMAgent
from slm_emergent_ai.memory.blackboard import BlackBoard
from slm_emergent_ai.models.loader import load_model, quantize_model
from slm_emergent_ai.controller.meta import MetaController
from slm_emergent_ai.eval.metrics import MetricsTracker


async def run_benchmark(model_path, quantize_type, num_agents=10, num_tokens=100, output_dir="./benchmark_results"):
    """
    ベンチマークを実行
    
    Parameters:
    -----------
    model_path: モデルパス
    quantize_type: 量子化タイプ
    num_agents: エージェント数
    num_tokens: 生成するトークン数
    output_dir: 出力ディレクトリ
    """
    print(f"Starting benchmark with {num_agents} agents, {num_tokens} tokens")
    print(f"Model: {model_path}, Quantization: {quantize_type}")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # プロセス情報を取得
    process = psutil.Process(os.getpid())
    
    # 初期メモリ使用量を記録
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # モデルをロード
    print("Loading model...")
    model_dict = load_model(model_path)
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    
    # モデルを量子化
    if quantize_type:
        print(f"Quantizing model to {quantize_type}...")
        model = quantize_model(model, quantize_type)
    
    # モデルロード後のメモリ使用量を記録
    after_load_memory = process.memory_info().rss / (1024 * 1024)  # MB
    model_memory = after_load_memory - initial_memory
    print(f"Model loaded. Memory usage: {after_load_memory:.2f} MB (model: {model_memory:.2f} MB)")
    
    # BlackBoardを初期化
    bb = BlackBoard()
    
    # λパラメータ
    lambda_params = {
        'λ_a': 0.3,
        'λ_c': 0.3,
        'λ_s': 0.1
    }
    
    # エージェントを作成
    print(f"Creating {num_agents} agents...")
    agents = []
    for i in range(num_agents):
        role = f"Agent-{i}"
        agent = SLMAgent(i, role, model, tokenizer, lambda_params)
        agents.append(agent)
    
    # エージェント作成後のメモリ使用量を記録
    after_agents_memory = process.memory_info().rss / (1024 * 1024)  # MB
    agents_memory = after_agents_memory - after_load_memory
    print(f"Agents created. Memory usage: {after_agents_memory:.2f} MB (agents: {agents_memory:.2f} MB)")
    
    # MetaControllerを初期化
    controller = MetaController(lambda_params, target_H=5.0)
    
    # メトリクストラッカーを初期化
    metrics = MetricsTracker()
    
    # 測定データ
    memory_usage = []
    cpu_usage = []
    token_counts = []
    times = []
    
    # 初期プロンプト
    prompt = "こんにちは、私はAIアシスタントです。今日は良い天気ですね。"
    
    # 開始時間を記録
    start_time = time.time()
    times.append(0.0)
    token_counts.append(0)
    memory_usage.append(after_agents_memory)
    cpu_usage.append(process.cpu_percent())
    
    # トークン生成ループ
    print(f"Generating {num_tokens} tokens...")
    current_prompt = prompt
    generated_tokens = 0
    
    for _ in tqdm(range(num_tokens)):
        # 各エージェントからトークンを生成
        agent_tasks = []
        for agent in agents:
            task = asyncio.create_task(agent.generate(current_prompt, bb))
            agent_tasks.append(task)
        
        # すべてのエージェントの応答を待つ
        responses = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # 有効な応答を集計
        valid_responses = [r for r in responses if isinstance(r, str)]
        if valid_responses:
            token = valid_responses[0]  # 最初のエージェントの応答を使用
            current_prompt += token
            generated_tokens += 1
        
        # 10トークンごとに測定
        if generated_tokens % 10 == 0:
            elapsed = time.time() - start_time
            times.append(elapsed)
            token_counts.append(generated_tokens)
            memory_usage.append(process.memory_info().rss / (1024 * 1024))
            cpu_usage.append(process.cpu_percent())
            
            # 速度を計算
            speed = generated_tokens / elapsed if elapsed > 0 else 0
            print(f"Generated {generated_tokens} tokens in {elapsed:.2f}s ({speed:.2f} tok/s)")
            
            # λパラメータを更新
            if hasattr(controller, 'update'):
                # 仮のエントロピー値
                entropy = np.random.uniform(4.0, 6.0)
                controller.update(entropy)
    
    # 終了時間を記録
    end_time = time.time()
    total_time = end_time - start_time
    
    # 最終測定
    memory_usage.append(process.memory_info().rss / (1024 * 1024))
    cpu_usage.append(process.cpu_percent())
    
    # 結果を表示
    final_memory = memory_usage[-1]
    memory_per_agent = (final_memory - initial_memory) / num_agents
    speed = generated_tokens / total_time if total_time > 0 else 0
    
    print("\nBenchmark Results:")
    print(f"Total tokens generated: {generated_tokens}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Speed: {speed:.2f} tok/s")
    print(f"Final memory usage: {final_memory:.2f} MB")
    print(f"Memory per agent: {memory_per_agent:.2f} MB")
    print(f"Average CPU usage: {np.mean(cpu_usage):.2f}%")
    
    # グラフを作成
    plt.figure(figsize=(12, 8))
    
    # メモリ使用量のプロット
    plt.subplot(2, 1, 1)
    plt.plot(times, memory_usage[:len(times)], 'b-', label='Memory Usage (MB)')
    plt.title('Memory Usage Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Memory (MB)')
    plt.grid(True)
    plt.legend()
    
    # CPU使用率のプロット
    plt.subplot(2, 1, 2)
    plt.plot(times, cpu_usage[:len(times)], 'r-', label='CPU Usage (%)')
    plt.title('CPU Usage Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('CPU (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # グラフを保存
    timestamp = int(time.time())
    plt.savefig(os.path.join(output_dir, f"bench_10agents_{timestamp}.png"))
    
    # 結果をテキストファイルに保存
    with open(os.path.join(output_dir, f"bench_10agents_{timestamp}.txt"), 'w') as f:
        f.write("Benchmark Results:\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Quantization: {quantize_type}\n")
        f.write(f"Number of agents: {num_agents}\n")
        f.write(f"Total tokens generated: {generated_tokens}\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write(f"Speed: {speed:.2f} tok/s\n")
        f.write(f"Initial memory usage: {initial_memory:.2f} MB\n")
        f.write(f"Final memory usage: {final_memory:.2f} MB\n")
        f.write(f"Memory per agent: {memory_per_agent:.2f} MB\n")
        f.write(f"Average CPU usage: {np.mean(cpu_usage):.2f}%\n")
    
    print(f"Results saved to {output_dir}")
    
    return {
        "tokens": generated_tokens,
        "time": total_time,
        "speed": speed,
        "memory": final_memory,
        "memory_per_agent": memory_per_agent,
        "cpu": np.mean(cpu_usage)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="10 Agents Benchmark")
    parser.add_argument("--model", type=str, default="gemma3:1b", help="Model path")
    parser.add_argument("--quantize", type=str, default="q4", help="Quantization type (q4, q8, f16, or none)")
    parser.add_argument("--agents", type=int, default=10, help="Number of agents")
    parser.add_argument("--tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--output", type=str, default="./benchmark_results", help="Output directory")
    
    args = parser.parse_args()
    
    # 量子化タイプを設定
    quantize_type = args.quantize if args.quantize.lower() != "none" else None
    
    # ベンチマークを実行
    asyncio.run(run_benchmark(args.model, quantize_type, args.agents, args.tokens, args.output))
