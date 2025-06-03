"""
PromptProcessor - Boidsルールをプロンプトエンジニアリングで実現するプロセッサ

従来のlogits操作の代わりに、プロンプトエンジニアリングを活用して
Boidsアルゴリズムの3つの基本ルール（整列・結合・分離）を実現する。
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import random

class BoidsPromptProcessor:
    """
    Boidsアルゴリズムをプロンプトエンジニアリングで実現するプロセッサ
    
    整列(Alignment)、結合(Cohesion)、分離(Separation)の3つの基本ルールを
    プロンプト操作によって実現する
    """
    def __init__(self, 
                 bb: 'BlackBoard',
                 λ: Dict[str, float] = None,
                 seed: Optional[int] = None):
        """
        BoidsPromptProcessorの初期化
        
        Parameters:
        -----------
        bb: BlackBoardインスタンス
        λ: 各ルールの重み係数（プロンプト内での強調度合いに影響）
        seed: 乱数シード（再現性のため）
        """
        self.bb = bb
        self.λ = λ if λ is not None else {'λ_a': 0.3, 'λ_c': 0.3, 'λ_s': 0.1}
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    async def process_prompt(self, base_prompt: str, agent_id: int, role: str, k: int = 16) -> str:
        """
        Boidsルールに基づいてプロンプトを加工
        
        Parameters:
        -----------
        base_prompt: 元のプロンプト
        agent_id: エージェントID
        role: エージェントの役割
        k: 取得する近傍メッセージ数
        
        Returns:
        --------
        Boidsルールを適用した後のプロンプト
        """
        # BlackBoardから近傍情報を取得
        messages = await self.bb.pull(k)
        
        # 各ルールの適用確率を計算（λ値に基づく）
        # 分離ルールの影響を意図的に下げる
        rule_weights = {
            'alignment': self.λ.get('λ_a', 0.3) * 1.5,  # 整列ルールを強調
            'cohesion': self.λ.get('λ_c', 0.3) * 1.5,   # 結合ルールを強調
            'separation': self.λ.get('λ_s', 0.1) * 0.5  # 分離ルールを弱める
        }
        
        # 合計が1になるように正規化
        total_weight = sum(rule_weights.values())
        rule_probs = {k: v/total_weight for k, v in rule_weights.items()}
        
        # ルールの選択（確率的に1つのルールを強調）
        rule_choices = list(rule_probs.keys())
        rule_values = list(rule_probs.values())
        
        # エージェントIDに基づいて異なるルールを優先（多様性のため）
        # エージェント1: 整列、エージェント2: 結合、エージェント3: 分離
        if agent_id % 3 == 1:
            primary_rule = 'alignment'  # 整列を優先
        elif agent_id % 3 == 2:
            primary_rule = 'cohesion'   # 結合を優先
        else:
            primary_rule = 'separation' # 分離を優先
        
        # 整列ルール（Alignment）のプロンプト部分を構築
        alignment_prompt = await self._build_alignment_prompt(messages)
        
        # 結合ルール（Cohesion）のプロンプト部分を構築
        cohesion_prompt = await self._build_cohesion_prompt()
        
        # 分離ルール（Separation）のプロンプト部分を構築
        separation_prompt = self._build_separation_prompt()
        
        # 主要ルールを強調したプロンプト構築
        boids_instructions = []
        
        if primary_rule == 'alignment':
            boids_instructions.append(f"【整列ルール】{alignment_prompt}")
            if rule_probs['cohesion'] > 0.2:
                boids_instructions.append(cohesion_prompt)
            if rule_probs['separation'] > 0.2:
                boids_instructions.append(separation_prompt)
        elif primary_rule == 'cohesion':
            boids_instructions.append(f"【結合ルール】{cohesion_prompt}")
            if rule_probs['alignment'] > 0.2:
                boids_instructions.append(alignment_prompt)
            if rule_probs['separation'] > 0.2:
                boids_instructions.append(separation_prompt)
        else:  # separation
            boids_instructions.append(f"【分離ルール】{separation_prompt}")
            if rule_probs['alignment'] > 0.2:
                boids_instructions.append(alignment_prompt)
            if rule_probs['cohesion'] > 0.2:
                boids_instructions.append(cohesion_prompt)
        
        # 最終的なプロンプトを構築
        final_prompt = f"""
{base_prompt}

あなたはエージェント{agent_id}で、役割は「{role}」です。
以下の指示に従って回答してください：

{' '.join(boids_instructions)}

回答：
"""
        
        return final_prompt
    
    async def _build_alignment_prompt(self, messages: List[str]) -> str:
        """
        整列ルール（Alignment）のプロンプト部分を構築
        
        Parameters:
        -----------
        messages: 近傍エージェントのメッセージリスト
        
        Returns:
        --------
        整列ルールを表現するプロンプト部分
        """
        if not messages:
            return "他のエージェントの発言はまだありません。独自の視点で考えてください。"
        
        # 近傍メッセージの要約
        summary = " ".join(messages[-3:])  # 最新の3つのメッセージを使用
        
        # 整列ルールの強度に応じてプロンプトの表現を調整
        strength = self.λ.get('λ_a', 0.3)
        
        alignment_prompts = [
            f"""
他のエージェントは次のように述べています：「{summary}」
これらの意見に強く同調し、同じ方向性で考えを発展させてください。
他のエージェントの考えを支持し、その延長線上にあるアイデアを提案してください。
""",
            f"""
他のエージェントは次のように述べています：「{summary}」
これらの意見を参考にして、同様の方向性で考えを述べてください。
他のエージェントの視点を取り入れ、協調的な意見を述べることが重要です。
""",
            f"""
他のエージェントは次のように述べています：「{summary}」
これらの意見も考慮しながら、回答を考えてください。
他のエージェントの考えを尊重し、部分的に取り入れてください。
"""
        ]
        
        if strength > 0.5:
            return alignment_prompts[0]
        elif strength > 0.3:
            return alignment_prompts[1]
        else:
            return alignment_prompts[2]
    
    async def _build_cohesion_prompt(self) -> str:
        """
        結合ルール（Cohesion）のプロンプト部分を構築
        
        Returns:
        --------
        結合ルールを表現するプロンプト部分
        """
        # BlackBoardからトピックサマリーを取得
        topic_summary = self.bb.get_topic_summary()
        
        # 結合ルールの強度に応じてプロンプトの表現を調整
        strength = self.λ.get('λ_c', 0.3)
        
        if not topic_summary:
            return "まだ明確なトピックは形成されていません。自由に考えを述べてください。"
        
        cohesion_prompts = [
            f"""
現在の議論の中心テーマは「{topic_summary}」です。
このテーマに密接に関連した内容を述べ、議論の一貫性を高めてください。
中心テーマから外れないよう注意し、議論を収束させる方向で考えてください。
""",
            f"""
現在の議論の中心テーマは「{topic_summary}」です。
このテーマを念頭に置きながら回答してください。
議論の中心から大きく逸脱しないよう意識してください。
""",
            f"""
現在の議論の中心テーマは「{topic_summary}」です。
このテーマも考慮に入れてください。
可能な範囲でこのテーマに関連する内容に触れてください。
"""
        ]
        
        if strength > 0.5:
            return cohesion_prompts[0]
        elif strength > 0.3:
            return cohesion_prompts[1]
        else:
            return cohesion_prompts[2]
    
    def _build_separation_prompt(self) -> str:
        """
        分離ルール（Separation）のプロンプト部分を構築
        
        Returns:
        --------
        分離ルールを表現するプロンプト部分
        """
        # 分離ルールの強度に応じてプロンプトの表現を調整
        strength = self.λ.get('λ_s', 0.1)
        
        # 多様性を促す視点のリスト
        perspectives = [
            "既存の枠組みを超えた発想で考えてみてください。",
            "他のエージェントとは明確に異なる視点を提供してください。",
            "独自の洞察や革新的なアイデアを積極的に提案してください。",
            "議論に多様性をもたらす対立的な視点を考えてください。",
            "既存の意見に対する建設的な反論や代替案を提案してください。"
        ]
        
        # ランダムに視点を選択
        selected_perspective = random.choice(perspectives)
        
        separation_prompts = [
            f"""
他のエージェントとは明確に異なる独自の視点を強く打ち出してください。
{selected_perspective}
議論の多様性を高めることが最も重要です。
あえて異なる立場から考え、新しい視点を導入してください。
""",
            f"""
他のエージェントとは異なる視点も取り入れてください。
{selected_perspective}
議論に新たな視点をもたらすことを意識してください。
""",
            f"""
必要に応じて、{selected_perspective.lower()}
多様な意見があることを示唆する内容も検討してください。
"""
        ]
        
        if strength > 0.3:
            return separation_prompts[0]
        elif strength > 0.1:
            return separation_prompts[1]
        else:
            return separation_prompts[2]
