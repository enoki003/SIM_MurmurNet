"""
テスト用スクリプト - プロンプトエンジニアリング方式のBoids制御の検証

プロンプトエンジニアリングによるBoids制御の効果を検証するためのスクリプト。
複数エージェントの協調動作をシミュレーションし、出力を評価する。
"""

import asyncio
import numpy as np
from typing import Dict, List, Any

from slm_emergent_ai.agents.core import SLMAgent, LLM
from slm_emergent_ai.memory.blackboard import BlackBoard
from slm_emergent_ai.boids.prompt_processor import BoidsPromptProcessor

async def run_simulation(num_agents: int = 3, turns_per_agent: int = 3):
    """
    複数エージェントによる会話シミュレーションを実行
    
    Parameters:
    -----------
    num_agents: エージェント数
    turns_per_agent: 各エージェントの発言回数
    """
    print(f"=== プロンプトエンジニアリング方式Boids制御シミュレーション開始 ===")
    print(f"エージェント数: {num_agents}, 各エージェントの発言回数: {turns_per_agent}")
    
    # BlackBoardの初期化
    bb = BlackBoard(mode="local")
    
    # モデルの初期化（実際の実装では実モデルを使用）
    # ここではモックモデルを使用
    class MockLLM:
        def generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> str:
            # プロンプトの内容に基づいて応答を生成するモック
            # 常に文字列を返すようにガード
            try:
                # 明示的なルールタグを検出
                if "【整列ルール】" in prompt:
                    return prompt + "\n\nはい、他のエージェントの意見に同意します。その方向性で考えを発展させると、人工知能の未来は協調的なシステムの発展にあると思います。他のエージェントが指摘したように、AIの進化は単独ではなく、相互作用によって加速するでしょう。"
                elif "【結合ルール】" in prompt:
                    return prompt + "\n\nご指摘の中心テーマについて考えを述べます。人工知能の未来に関して、特に重要なのは倫理的な発展と人間との共存です。このテーマに関連して、AIの意思決定の透明性と説明可能性が今後ますます重要になるでしょう。"
                elif "【分離ルール】" in prompt:
                    return prompt + "\n\n他とは異なる視点から考えると、人工知能の未来は必ずしも人間型の知能を目指すものではないかもしれません。むしろ、人間の認知とは全く異なる形の知能が発展し、私たちの想像を超える問題解決方法を生み出す可能性があります。"
                # 通常の検出ロジック（より詳細な分岐）
                elif "同じ方向性" in prompt or "同調" in prompt or "協調的" in prompt:
                    return prompt + "\n\nはい、他のエージェントの意見に同意します。その方向性で考えを発展させると、人工知能の未来は協調的なシステムの発展にあると思います。他のエージェントが指摘したように、AIの進化は単独ではなく、相互作用によって加速するでしょう。"
                elif "中心テーマ" in prompt or "一貫性" in prompt or "収束" in prompt:
                    return prompt + "\n\nご指摘の中心テーマについて考えを述べます。人工知能の未来に関して、特に重要なのは倫理的な発展と人間との共存です。このテーマに関連して、AIの意思決定の透明性と説明可能性が今後ますます重要になるでしょう。"
                elif "異なる視点" in prompt or "多様性" in prompt or "革新的" in prompt:
                    return prompt + "\n\n他とは異なる視点から考えると、人工知能の未来は必ずしも人間型の知能を目指すものではないかもしれません。むしろ、人間の認知とは全く異なる形の知能が発展し、私たちの想像を超える問題解決方法を生み出す可能性があります。"
                else:
                    return prompt + "\n\n人工知能の未来について考えると、技術的な進歩と社会的な影響の両面から検討する必要があります。特に注目すべきは、AIの自律性の向上と、それに伴う倫理的・法的課題です。また、AIと人間の協働の形も今後大きく変化していくでしょう。"
            except Exception as e:
                print(f"MockLLM生成エラー: {e}")
                # エラー時はデフォルト応答を返す
                return prompt + "\n\nデフォルト応答: 人工知能の未来は技術と倫理のバランスが重要です。"
    
    model = MockLLM()
    
    # エージェントの初期化
    agents = []
    roles = ["質問者", "回答者", "批評家", "統合者", "発散者"]
    
    for i in range(num_agents):
        # λパラメータをエージェントごとに少し変える
        λ = {
            'λ_a': 0.3 + np.random.uniform(-0.1, 0.1),
            'λ_c': 0.3 + np.random.uniform(-0.1, 0.1),
            'λ_s': 0.1 + np.random.uniform(-0.05, 0.15)
        }
        
        agent = SLMAgent(
            id=i+1,
            role=roles[i % len(roles)],
            model=model,
            λ=λ
        )
        
        # プロンプトプロセッサを設定
        prompt_processor = BoidsPromptProcessor(bb, λ)
        agent.set_prompt_processor(prompt_processor)
        
        agents.append(agent)
    
    # 初期プロンプト
    initial_prompt = "人工知能の未来について議論してください。"
    
    # シミュレーション実行
    for turn in range(turns_per_agent):
        print(f"\n--- ターン {turn+1} ---")
        
        for i, agent in enumerate(agents):
            try:
                # エージェントの発言を生成
                response = await agent.generate(initial_prompt, bb)
                
                # BlackBoardに情報をプッシュ
                await bb.push(agent.id, response)
                
                # 応答の表示（最大100文字）
                display_response = response[:100] + "..." if len(response) > 100 else response
                print(f"エージェント{agent.id}（{agent.role}）: {display_response}")
                
                # λパラメータの表示
                print(f"  λパラメータ: λ_a={agent.λ['λ_a']:.2f}, λ_c={agent.λ['λ_c']:.2f}, λ_s={agent.λ['λ_s']:.2f}")
            except Exception as e:
                print(f"エージェント{agent.id}の処理中にエラーが発生: {e}")
                # エラー時もBlackBoardに情報をプッシュ（会話の流れを維持）
                await bb.push(agent.id, f"エラーが発生しましたが、議論を続けます。人工知能の未来について考えると...")
    
    # トピックサマリーの表示
    topic_summary = bb.get_topic_summary()
    print(f"\n最終トピックサマリー: {topic_summary}")
    
    print("\n=== シミュレーション終了 ===")

if __name__ == "__main__":
    asyncio.run(run_simulation())
