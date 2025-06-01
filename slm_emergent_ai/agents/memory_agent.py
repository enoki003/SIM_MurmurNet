"""
Memory Agent - 記憶を持つエージェントの実装

BlackBoardとRAGを組み合わせて長期記憶と短期記憶を持つエージェントの実装。
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
import numpy as np

from ..agents.core import SLMAgent, LLM
from ..agents.rag import RAGAgent

class MemoryAgent:
    """
    長期記憶と短期記憶を持つエージェント
    
    BlackBoardを短期記憶として、RAGを長期記憶として使用する
    """
    def __init__(self, id: int, role: str, model: LLM, λ: Dict[str, float], 
                 rag_agent: Optional[RAGAgent] = None,
                 memory_threshold: float = 0.7):
        """
        MemoryAgentの初期化
        
        Parameters:
        -----------
        id: エージェントID
        role: エージェントの役割
        model: 言語モデル
        λ: Boidsルールの重み係数
        rag_agent: RAGエージェント（長期記憶用）
        memory_threshold: 長期記憶に保存する重要度の閾値
        """
        self.id = id
        self.role = role
        self.model = model
        self.λ = λ
        self.rag_agent = rag_agent
        self.memory_threshold = memory_threshold
        self.cache = {}
        self.short_term_memory = []  # 短期記憶
        self.importance_history = []  # 重要度の履歴
    
    def _calculate_importance(self, text: str) -> float:
        """
        テキストの重要度を計算
        
        Parameters:
        -----------
        text: テキスト
        
        Returns:
        --------
        重要度スコア (0.0-1.0)
        """
        # 実際の実装では、テキストの情報量や新規性などに基づいて重要度を計算
        # ここではダミー実装
        import random
        return random.uniform(0.0, 1.0)
    
    async def _store_to_long_term_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        長期記憶にテキストを保存
        
        Parameters:
        -----------
        text: テキスト
        metadata: メタデータ
        
        Returns:
        --------
        ドキュメントID
        """
        if self.rag_agent:
            metadata = metadata or {}
            metadata["agent_id"] = self.id
            metadata["role"] = self.role
            metadata["timestamp"] = asyncio.get_event_loop().time()
            
            return self.rag_agent.add_to_memory(text, metadata)
        return ""
    
    def _update_short_term_memory(self, text: str):
        """
        短期記憶を更新
        
        Parameters:
        -----------
        text: テキスト
        """
        self.short_term_memory.append(text)
        
        # 短期記憶のサイズを制限
        if len(self.short_term_memory) > 10:
            self.short_term_memory.pop(0)
    
    async def generate(self, prompt: str, bb: 'BlackBoard') -> str:
        """
        Boidsルールと記憶を使用してテキスト生成
        
        Parameters:
        -----------
        prompt: 入力プロンプト
        bb: BlackBoardインスタンス
        
        Returns:
        --------
        生成されたトークン
        """
        # BlackBoardから近傍情報を取得
        from ..agents.core import BoidsCtx
        ctx = BoidsCtx(bb.pull(k=16), bb.summary_vec)
        
        # 長期記憶から関連情報を取得
        long_term_context = ""
        if self.rag_agent:
            # プロンプトに基づいて関連情報を検索
            results = self.rag_agent.backend.search(prompt, top_k=3)
            if results:
                long_term_context = "長期記憶:\n" + "\n".join([f"- {r['text']}" for r in results])
        
        # 短期記憶を追加
        short_term_context = ""
        if self.short_term_memory:
            short_term_context = "短期記憶:\n" + "\n".join([f"- {m}" for m in self.short_term_memory])
        
        # コンテキストを組み合わせる
        combined_prompt = prompt
        if long_term_context:
            combined_prompt = f"{long_term_context}\n\n{combined_prompt}"
        if short_term_context:
            combined_prompt = f"{short_term_context}\n\n{combined_prompt}"
        
        # モデルの推論実行
        logits = self.model.forward(combined_prompt)
        
        # Boidsルールを適用
        from ..boids.rules import apply_boids_rules
        logits = apply_boids_rules(logits, ctx.neighbor_vecs, ctx.summary_vec, self.λ)
        
        # トークンのサンプリング
        from ..agents.core import sample_top_p
        token = sample_top_p(logits)
        
        # トークンをデコード
        token_str = self.model.tokenizer.decode([token])
        
        return token_str
    
    async def process_and_memorize(self, text: str, bb: 'BlackBoard'):
        """
        テキストを処理し、必要に応じて記憶に保存
        
        Parameters:
        -----------
        text: テキスト
        bb: BlackBoardインスタンス
        """
        # 短期記憶に追加
        self._update_short_term_memory(text)
        
        # 重要度を計算
        importance = self._calculate_importance(text)
        self.importance_history.append(importance)
        
        # 重要度が閾値を超える場合、長期記憶に保存
        if importance > self.memory_threshold:
            await self._store_to_long_term_memory(text, {"importance": importance})
        
        # BlackBoardに情報をプッシュ
        await bb.push(self.id, text)
    
    async def run_conversation(self, initial_prompt: str, bb: 'BlackBoard', max_turns: int = 10) -> List[str]:
        """
        会話を実行
        
        Parameters:
        -----------
        initial_prompt: 初期プロンプト
        bb: BlackBoardインスタンス
        max_turns: 最大ターン数
        
        Returns:
        --------
        会話履歴
        """
        conversation = [initial_prompt]
        current_prompt = initial_prompt
        
        # 初期プロンプトを処理
        await self.process_and_memorize(initial_prompt, bb)
        
        for _ in range(max_turns):
            # トークン生成
            token = await self.generate(current_prompt, bb)
            
            # 会話に追加
            conversation.append(token)
            current_prompt += token
            
            # テキストを処理し、記憶に保存
            await self.process_and_memorize(token, bb)
        
        return conversation
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        記憶の統計情報を取得
        
        Returns:
        --------
        統計情報
        """
        return {
            "short_term_memory_size": len(self.short_term_memory),
            "importance_mean": np.mean(self.importance_history) if self.importance_history else 0.0,
            "importance_std": np.std(self.importance_history) if self.importance_history else 0.0,
            "long_term_memory_available": self.rag_agent is not None
        }
