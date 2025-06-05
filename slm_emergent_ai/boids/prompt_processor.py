"""
PromptProcessor - Boidsルールをプロンプトエンジニアリングで実現するプロセッサ

従来のlogits操作の代わりに、プロンプトエンジニアリングを活用して
Boidsアルゴリズムの3つの基本ルール（整列・結合・分離）を実現する。
"""

from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING
import numpy as np
import random

if TYPE_CHECKING:
    from ..memory.blackboard import BlackBoard

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
        """
        # BlackBoardから近傍情報を取得
        try:
            messages = await self.bb.pull_messages_raw(k)
            print(f"[DEBUG] BoidsPromptProcessor: Agent {agent_id} ({role}) received {len(messages)} raw messages")
        except AttributeError:
            messages = await self.bb.pull(k)
            print(f"[DEBUG] BoidsPromptProcessor: Agent {agent_id} ({role}) received {len(messages)} messages")
        
        # メッセージのテキスト部分を抽出
        speaker_info = []
        for msg in messages:
            if isinstance(msg, dict):
                text = msg.get('text', str(msg))
                agent_name = msg.get('agent_name', msg.get('role', 'Unknown'))
                speaker_info.append(f"{agent_name}: {text}")
            elif isinstance(msg, str):
                speaker_info.append(f"発言者不明: {msg}")
            else:
                speaker_info.append(f"発言者不明: {str(msg)}")
        
        # エージェントIDに基づいて異なるルールを優先（多様性のため）
        if agent_id % 3 == 1:
            primary_rule = 'alignment'  # 整列を優先
        elif agent_id % 3 == 2:
            primary_rule = 'cohesion'   # 結合を優先
        else:
            primary_rule = 'separation' # 分離を優先
        
        # 各ルールのプロンプト部分を構築
        alignment_prompt = self._build_alignment_prompt(speaker_info)
        cohesion_prompt = self._build_cohesion_prompt(speaker_info)
        separation_prompt = self._build_separation_prompt(role)
        
        # 主要ルールを強調したプロンプト構築
        boids_instructions = []
        
        if primary_rule == 'alignment':
            boids_instructions.append(f"【整列ルール】{alignment_prompt}")
            boids_instructions.append(f"【結合ルール】{cohesion_prompt}")
        elif primary_rule == 'cohesion':
            boids_instructions.append(f"【結合ルール】{cohesion_prompt}")
            boids_instructions.append(f"【整列ルール】{alignment_prompt}")
        else:  # separation
            boids_instructions.append(f"【分離ルール】{separation_prompt}")
            boids_instructions.append(f"【整列ルール】{alignment_prompt}")
        
        # 役割固有の指示を追加
        role_instruction = self._build_role_instruction(role)
        
        # 最終的なプロンプトを構築
        enhanced_prompt = f"""{role_instruction}

{chr(10).join(boids_instructions)}

{base_prompt}

前述の指示に基づいて、あなたの役割({role})として自然で建設的な応答をしてください。"""
        
        return enhanced_prompt

    def _build_alignment_prompt(self, speaker_info: List[str]) -> str:
        """整列ルール（Alignment）のプロンプトを構築"""
        if not speaker_info:
            return "最近の議論の流れを考慮して発言してください。"
        
        recent_messages = speaker_info[-3:] if len(speaker_info) >= 3 else speaker_info
        prompt = "以下の最近の発言を参考に、議論の流れに沿った応答をしてください：\n"
        for msg in recent_messages:
            prompt += f"- {msg}\n"
        prompt += "上記の発言の方向性や論調を意識して、建設的に議論を続けてください。"
        return prompt

    def _build_cohesion_prompt(self, speaker_info: List[str]) -> str:
        """結合ルール（Cohesion）のプロンプトを構築"""
        if not speaker_info:
            return "議論全体の目標に向かって発言してください。"
        
        speakers = set()
        for msg in speaker_info:
            if ':' in msg:
                speaker = msg.split(':', 1)[0]
                speakers.add(speaker)
        
        prompt = f"これまでの議論には{len(speakers)}名が参加しており、以下のような流れです：\n"
        summary_messages = speaker_info[-5:] if len(speaker_info) >= 5 else speaker_info
        for msg in summary_messages:
            prompt += f"- {msg}\n"
        prompt += "議論全体の一貫性を保ち、共通の理解に向けて貢献してください。"
        return prompt

    def _build_separation_prompt(self, role: str) -> str:
        """分離ルール（Separation）のプロンプトを構築"""
        separation_instructions = {
            "質問者": "他の参加者とは違う角度から、新しい疑問や問題提起をしてください。",
            "回答者": "他の発言とは異なる具体的で実用的な解決策を提示してください。",
            "批評者": "これまでの議論では触れられていない問題点や改善点を指摘してください。",
            "批判者": "これまでの議論では触れられていない問題点や改善点を指摘してください。",
        }
        instruction = separation_instructions.get(role, "あなた独自の視点で、他とは異なる貢献をしてください。")
        return f"あなたの役割({role})の特性を活かし、{instruction}"

    def _build_role_instruction(self, role: str) -> str:
        """役割固有の基本指示を構築"""
        role_instructions = {
            "質問者": """あなたは好奇心旺盛な質問者です。
- 議論を深める質問を投げかけてください
- 「なぜ？」「どのように？」「他には？」といった探求的な姿勢を示してください
- 新しい視点や角度から問題を捉えてください""",
            
            "回答者": """あなたは知識豊富な回答者です。
- 具体的で実用的な回答や解決策を提供してください
- 事例や経験に基づいた説明を心がけてください
- 分かりやすく建設的な情報を提供してください""",
            
            "批評者": """あなたは冷静な批評者です。
- 客観的な視点で議論を分析してください
- 長所と短所をバランスよく評価してください
- 改善点や代替案を建設的に提示してください""",
            
            "批判者": """あなたは冷静な批評者です。
- 客観的な視点で議論を分析してください
- 長所と短所をバランスよく評価してください
- 改善点や代替案を建設的に提示してください""",
        }
        return role_instructions.get(role, "あなたは議論に参加する一員として、建設的な貢献をしてください。")
