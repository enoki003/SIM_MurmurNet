"""
ベンチマークモジュール - システムの性能評価

TriviaQAなどのデータセットを使用してシステムの性能を評価するベンチマークモジュール。
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

class TriviaQARunner:
    """
    TriviaQAデータセットを使用したベンチマーク実行クラス
    """
    def __init__(self, data_path: str = "data/triviaqa-sample.json", 
                 output_dir: str = "./benchmark_results"):
        """
        TriviaQARunnerの初期化
        
        Parameters:
        -----------
        data_path: TriviaQAデータセットのパス
        output_dir: 結果出力ディレクトリ
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.questions = []
        self._load_data()
        os.makedirs(output_dir, exist_ok=True)
    
    def _load_data(self):
        """データセットの読み込み"""
        try:
            if os.path.exists(self.data_path):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    self.questions = data
                elif isinstance(data, dict) and 'data' in data:
                    self.questions = data['data']
                else:
                    print(f"Unknown data format in {self.data_path}")
                    self.questions = []
                
                print(f"Loaded {len(self.questions)} questions from {self.data_path}")
            else:
                print(f"Data file not found: {self.data_path}")
                self.questions = []
        except Exception as e:
            print(f"Error loading data: {e}")
            self.questions = []
    
    async def run_benchmark(self, agents: List[Any], bb: 'BlackBoard', 
                           num_questions: int = 10, timeout: int = 60) -> Dict[str, Any]:
        """
        ベンチマークを実行
        
        Parameters:
        -----------
        agents: エージェントのリスト
        bb: BlackBoardインスタンス
        num_questions: 使用する質問数
        timeout: 各質問のタイムアウト (秒)
        
        Returns:
        --------
        ベンチマーク結果
        """
        if not self.questions:
            return {"error": "No questions loaded"}
        
        # 使用する質問を選択
        selected_questions = self.questions[:min(num_questions, len(self.questions))]
        
        results = []
        correct_count = 0
        total_time = 0
        
        print(f"Running benchmark with {len(agents)} agents on {len(selected_questions)} questions...")
        
        for i, question in enumerate(tqdm(selected_questions)):
            # BlackBoardをクリア
            if hasattr(bb, 'clear_all'):
                bb.clear_all()
            
            q_text = question.get('question', '')
            answers = question.get('answers', [])
            correct_answer = answers[0] if answers else ""
            
            # プロンプトを作成
            prompt = f"質問: {q_text}\n回答: "
            
            # 開始時間を記録
            start_time = time.time()
            
            # エージェントに質問を投げる
            try:
                # タイムアウト付きで実行
                agent_tasks = []
                for agent in agents:
                    task = asyncio.create_task(agent.generate(prompt, bb))
                    agent_tasks.append(task)
                
                # すべてのエージェントの応答を待つ
                responses = await asyncio.gather(*agent_tasks, return_exceptions=True)
                
                # 最終的な回答を集約
                final_answer = " ".join([r for r in responses if isinstance(r, str)])
                
                # 経過時間を記録
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                
                # 正解かどうかを判定
                is_correct = any(ans.lower() in final_answer.lower() for ans in answers)
                if is_correct:
                    correct_count += 1
                
                # 結果を記録
                results.append({
                    "question_id": i,
                    "question": q_text,
                    "correct_answers": answers,
                    "agent_answer": final_answer,
                    "is_correct": is_correct,
                    "time_taken": elapsed_time
                })
                
            except asyncio.TimeoutError:
                print(f"Question {i} timed out")
                results.append({
                    "question_id": i,
                    "question": q_text,
                    "correct_answers": answers,
                    "agent_answer": "TIMEOUT",
                    "is_correct": False,
                    "time_taken": timeout
                })
                total_time += timeout
        
        # 結果をまとめる
        accuracy = correct_count / len(selected_questions) if selected_questions else 0
        avg_time = total_time / len(selected_questions) if selected_questions else 0
        
        benchmark_results = {
            "total_questions": len(selected_questions),
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "average_time": avg_time,
            "total_time": total_time,
            "detailed_results": results
        }
        
        # 結果を保存
        timestamp = int(time.time())
        output_path = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print(f"Benchmark completed. Accuracy: {accuracy:.2f}, Avg time: {avg_time:.2f}s")
        print(f"Results saved to {output_path}")
        
        return benchmark_results
    
    def analyze_results(self, results_path: Optional[str] = None) -> Dict[str, Any]:
        """
        ベンチマーク結果を分析
        
        Parameters:
        -----------
        results_path: 結果ファイルのパス (Noneの場合は最新の結果を使用)
        
        Returns:
        --------
        分析結果
        """
        # 結果ファイルを取得
        if results_path is None:
            result_files = [f for f in os.listdir(self.output_dir) if f.startswith("benchmark_results_")]
            if not result_files:
                return {"error": "No benchmark results found"}
            results_path = os.path.join(self.output_dir, sorted(result_files)[-1])
        
        # 結果を読み込む
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception as e:
            return {"error": f"Error loading results: {e}"}
        
        # 詳細結果をDataFrameに変換
        df = pd.DataFrame(results.get("detailed_results", []))
        
        # 分析結果
        analysis = {
            "accuracy": results.get("accuracy", 0),
            "average_time": results.get("average_time", 0),
            "total_questions": results.get("total_questions", 0),
            "correct_answers": results.get("correct_answers", 0),
            "time_stats": {
                "min": df["time_taken"].min() if not df.empty else 0,
                "max": df["time_taken"].max() if not df.empty else 0,
                "median": df["time_taken"].median() if not df.empty else 0,
                "std": df["time_taken"].std() if not df.empty else 0
            },
            "error_analysis": {
                "timeout_count": len(df[df["agent_answer"] == "TIMEOUT"]) if not df.empty else 0,
                "incorrect_count": len(df[~df["is_correct"] & (df["agent_answer"] != "TIMEOUT")]) if not df.empty else 0
            }
        }
        
        return analysis


class PerformanceBenchmark:
    """
    システム全体のパフォーマンスベンチマーク
    """
    def __init__(self, output_dir: str = "./benchmark_results"):
        """
        PerformanceBenchmarkの初期化
        
        Parameters:
        -----------
        output_dir: 結果出力ディレクトリ
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    async def run_speed_benchmark(self, agents: List[Any], bb: 'BlackBoard', 
                                 prompt: str = "こんにちは、今日の天気について教えてください。",
                                 num_tokens: int = 100,
                                 repeat: int = 5) -> Dict[str, Any]:
        """
        生成速度のベンチマークを実行
        
        Parameters:
        -----------
        agents: エージェントのリスト
        bb: BlackBoardインスタンス
        prompt: 入力プロンプト
        num_tokens: 生成するトークン数
        repeat: 繰り返し回数
        
        Returns:
        --------
        ベンチマーク結果
        """
        speeds = []
        token_counts = []
        times = []
        
        print(f"Running speed benchmark with {len(agents)} agents, {num_tokens} tokens, {repeat} repeats...")
        
        for i in range(repeat):
            # BlackBoardをクリア
            if hasattr(bb, 'clear_all'):
                bb.clear_all()
            
            # 開始時間を記録
            start_time = time.time()
            
            # エージェントに生成させる
            generated_tokens = 0
            current_prompt = prompt
            
            for _ in range(num_tokens):
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
            
            # 経過時間を記録
            elapsed_time = time.time() - start_time
            
            # 速度を計算 (tok/s)
            speed = generated_tokens / elapsed_time if elapsed_time > 0 else 0
            
            speeds.append(speed)
            token_counts.append(generated_tokens)
            times.append(elapsed_time)
            
            print(f"Run {i+1}/{repeat}: {speed:.2f} tok/s ({generated_tokens} tokens in {elapsed_time:.2f}s)")
        
        # 結果をまとめる
        # 幾何平均を計算
        geometric_mean = np.exp(np.mean(np.log(np.array(speeds) + 1e-10))) - 1e-10
        
        benchmark_results = {
            "speeds": speeds,
            "token_counts": token_counts,
            "times": times,
            "arithmetic_mean": np.mean(speeds),
            "geometric_mean": geometric_mean,
            "median": np.median(speeds),
            "min": np.min(speeds),
            "max": np.max(speeds),
            "std": np.std(speeds)
        }
        
        # 結果を保存
        timestamp = int(time.time())
        output_path = os.path.join(self.output_dir, f"speed_benchmark_{timestamp}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print(f"Speed benchmark completed. Geometric mean: {geometric_mean:.2f} tok/s")
        print(f"Results saved to {output_path}")
        
        return benchmark_results
    
    async def run_memory_benchmark(self, agents: List[Any], bb: 'BlackBoard',
                                  num_agents_range: List[int] = [1, 3, 5, 10, 20, 30],
                                  duration: int = 10) -> Dict[str, Any]:
        """
        メモリ使用量のベンチマークを実行
        
        Parameters:
        -----------
        agents: エージェントのリスト
        bb: BlackBoardインスタンス
        num_agents_range: テストするエージェント数のリスト
        duration: 各テストの実行時間 (秒)
        
        Returns:
        --------
        ベンチマーク結果
        """
        import psutil
        
        process = psutil.Process(os.getpid())
        results = []
        
        print(f"Running memory benchmark with varying agent counts...")
        
        for num_agents in num_agents_range:
            if num_agents > len(agents):
                print(f"Skipping {num_agents} agents (only {len(agents)} available)")
                continue
            
            print(f"Testing with {num_agents} agents for {duration} seconds...")
            
            # BlackBoardをクリア
            if hasattr(bb, 'clear_all'):
                bb.clear_all()
            
            # 初期メモリ使用量を記録
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # 指定した数のエージェントを使用
            test_agents = agents[:num_agents]
            
            # プロンプト
            prompt = "こんにちは、私はAIアシスタントです。"
            
            # 開始時間を記録
            start_time = time.time()
            
            # 指定時間だけ実行
            generated_tokens = 0
            current_prompt = prompt
            
            while time.time() - start_time < duration:
                agent_tasks = []
                for agent in test_agents:
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
            
            # 最終メモリ使用量を記録
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_increase = final_memory - initial_memory
            
            # 結果を記録
            results.append({
                "num_agents": num_agents,
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": memory_increase,
                "memory_per_agent_mb": memory_increase / num_agents if num_agents > 0 else 0,
                "generated_tokens": generated_tokens,
                "duration": duration
            })
            
            print(f"{num_agents} agents: {memory_increase:.2f} MB increase ({memory_increase/num_agents:.2f} MB/agent)")
        
        # 結果をまとめる
        benchmark_results = {
            "detailed_results": results,
            "summary": {
                "max_memory_increase": max([r["memory_increase_mb"] for r in results]) if results else 0,
                "avg_memory_per_agent": np.mean([r["memory_per_agent_mb"] for r in results]) if results else 0
            }
        }
        
        # 結果を保存
        timestamp = int(time.time())
        output_path = os.path.join(self.output_dir, f"memory_benchmark_{timestamp}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print(f"Memory benchmark completed.")
        print(f"Results saved to {output_path}")
        
        return benchmark_results
