"""
MetaController - システム全体の制御と調整を行うコントローラー

λ自動調整にPID制御と強化学習(Q-learning)を併用。
スパイク検知でEarlyStop機能を提供。
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..memory.blackboard import BlackBoard

class MetaController:
    """
    システム全体のパラメータ調整と制御を行うメタコントローラー
    
    PID制御と強化学習を組み合わせてλパラメータを自動調整する
    """
    def __init__(self, bb: 'BlackBoard', target_H: float = 5.0, 
                 Kp: float = 0.1, Ki: float = 0.01, Kd: float = 0.05):
        """
        MetaControllerの初期化
        
        Parameters:
        -----------
        bb: BlackBoardインスタンス
        target_H: 目標エントロピー値
        Kp, Ki, Kd: PID制御のゲイン係数
        """
        self.bb = bb
        self.target_H = target_H
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        # PID制御の状態変数
        self.prev_err = 0.0
        self.int = 0.0
        
        # Q学習のパラメータ
        self.Q = {}  # 状態-行動価値関数
        self.alpha = 0.1  # 学習率
        self.gamma = 0.9  # 割引率
        self.epsilon = 0.1  # 探索率
        
        # 監視用変数
        self.early_stop = False
        self.last_update = time.time()
    
    async def adjust_lambda(self, stats: Dict[str, float]) -> Dict[str, float]:
        """
        統計情報に基づいてλパラメータを調整
        
        Parameters:
        -----------
        stats: 統計情報 (entropy, vdi, scd, fcr, speed)
        
        Returns:
        --------
        調整後のλパラメータ
        """
        # 現在のλパラメータを取得
        λ_c = await self.bb.get_param('λ_c', 0.3)
        λ_a = await self.bb.get_param('λ_a', 0.3)
        λ_s = await self.bb.get_param('λ_s', 0.1)
        
        # エントロピー誤差を計算
        err = stats['entropy'] - self.target_H
        dt = time.time() - self.last_update
        self.last_update = time.time()
        
        # PID制御による調整量を計算
        self.int += err * dt
        diff = (err - self.prev_err) / max(dt, 0.001)
        self.prev_err = err
        
        λ_adj = self.Kp * err + self.Ki * self.int + self.Kd * diff
        
        # λパラメータを調整
        λ_c_new = self._clamp(λ_c + λ_adj, 0.0, 1.0)
        
        # 安定条件: λ_a + λ_c > λ_s を維持
        if λ_a + λ_c_new <= λ_s:
            # λ_sを減衰させる
            λ_s_new = self._clamp(λ_a + λ_c_new - 0.05, 0.0, 0.7)
        else:
            λ_s_new = λ_s
        
        # Q学習による調整（状態を離散化）
        state = await self._discretize_state(stats)
        action = self._select_action(state)
        
        # VDIに応じたλ_s自動調整ロジック（強化版）
        vdi_value = stats.get('vdi', 0)
        
        # 行動に基づいてλを微調整
        if action == 0:  # λ_c増加
            λ_c_new = self._clamp(λ_c_new + 0.02, 0.0, 1.0)
        elif action == 1:  # λ_c減少
            λ_c_new = self._clamp(λ_c_new - 0.02, 0.0, 1.0)
        elif action == 2:  # λ_a増加
            λ_a = self._clamp(λ_a + 0.02, 0.0, 1.0)
        elif action == 3:  # λ_a減少
            λ_a = self._clamp(λ_a - 0.02, 0.0, 1.0)
        
        # VDIベースのλ_s自動調整（追加ロジック）
        if vdi_value < 0.2:  # 非常に低い多様性
            λ_s_new = self._clamp(λ_s + 0.08, 0.0, 0.5)
            print(f"[CONTROLLER] Very low VDI ({vdi_value:.3f}) - Significantly increasing λ_s to {λ_s_new:.3f}")
        elif vdi_value < 0.4:  # 低い多様性
            λ_s_new = self._clamp(λ_s + 0.04, 0.0, 0.5)
            print(f"[CONTROLLER] Low VDI ({vdi_value:.3f}) - Increasing λ_s to {λ_s_new:.3f}")
        elif vdi_value > 0.8:  # 非常に高い多様性
            λ_s_new = self._clamp(λ_s - 0.06, 0.05, 0.5)
            print(f"[CONTROLLER] Very high VDI ({vdi_value:.3f}) - Significantly decreasing λ_s to {λ_s_new:.3f}")
        elif vdi_value > 0.6:  # 高い多様性
            λ_s_new = self._clamp(λ_s - 0.03, 0.05, 0.5)
            print(f"[CONTROLLER] High VDI ({vdi_value:.3f}) - Decreasing λ_s to {λ_s_new:.3f}")
        
        # 安定条件の再確認と調整
        if λ_a + λ_c_new <= λ_s_new:
            # λ_sが大きすぎる場合の調整
            if λ_s_new > 0.3:
                λ_s_new = self._clamp(λ_a + λ_c_new - 0.05, 0.05, 0.3)
                print(f"[CONTROLLER] Stability constraint enforced - Reducing λ_s to {λ_s_new:.3f}")
            else:
                # λ_cを増加させて安定性を確保
                λ_c_new = self._clamp(λ_s_new + 0.05 - λ_a, 0.1, 1.0)
                print(f"[CONTROLLER] Stability constraint enforced - Increasing λ_c to {λ_c_new:.3f}")
        
        # スパイク検知によるEarlyStop
        if abs(err) > 5.0 or stats.get('speed', 0) < 0.5 * await self.bb.get_param('base_speed', 10.0):
            self.early_stop = True
        
        # BlackBoardにパラメータを設定
        await self.bb.set_param('λ_c', λ_c_new)
        await self.bb.set_param('λ_a', λ_a)
        await self.bb.set_param('λ_s', λ_s_new)
        
        # 調整後のλパラメータを返す
        return {
            'λ_c': λ_c_new,
            'λ_a': λ_a,
            'λ_s': λ_s_new
        }
    
    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """値を指定範囲に制限"""
        return max(min_val, min(max_val, value))
    
    async def _discretize_state(self, stats: Dict[str, float]) -> str:
        """
        連続的な状態を離散化
        
        Parameters:
        -----------
        stats: 統計情報
        
        Returns:
        --------
        離散化された状態を表す文字列
        """
        # エントロピーを離散化 (低/中/高)
        entropy = stats.get('entropy', 0)
        if entropy < self.target_H - 1:
            e_state = 'low'
        elif entropy > self.target_H + 1:
            e_state = 'high'
        else:
            e_state = 'mid'
        
        # VDIを離散化 (低/中/高)
        vdi = stats.get('vdi', 0)
        if vdi < 0.3:
            v_state = 'low'
        elif vdi > 0.7:
            v_state = 'high'
        else:
            v_state = 'mid'
        
        # 速度を離散化 (低/中/高)
        speed = stats.get('speed', 0)
        base_speed = await self.bb.get_param('base_speed', 10.0)
        if speed < 0.7 * base_speed:
            s_state = 'low'
        elif speed > 1.3 * base_speed:
            s_state = 'high'
        else:
            s_state = 'mid'
        
        return f"{e_state}_{v_state}_{s_state}"
    
    def _select_action(self, state: str) -> int:
        """
        Q学習に基づいて行動を選択
        
        Parameters:
        -----------
        state: 現在の状態
        
        Returns:
        --------
        選択された行動のインデックス
        """
        # ε-greedy方策
        if np.random.random() < self.epsilon:
            # ランダムに行動を選択 (探索)
            return np.random.randint(0, 4)
        
        # 状態が未知の場合は初期化
        if state not in self.Q:
            self.Q[state] = np.zeros(4)
        
        # 最大Q値を持つ行動を選択 (活用)
        return np.argmax(self.Q[state])
    
    def update_Q(self, state: str, action: int, reward: float, next_state: str):
        """
        Q値を更新
        
        Parameters:
        -----------
        state: 現在の状態
        action: 実行した行動
        reward: 得られた報酬
        next_state: 次の状態
        """
        # 状態が未知の場合は初期化
        if state not in self.Q:
            self.Q[state] = np.zeros(4)
        if next_state not in self.Q:
            self.Q[next_state] = np.zeros(4)
        
        # Q値を更新
        max_next_Q = np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (reward + self.gamma * max_next_Q - self.Q[state][action])
    
    async def calculate_reward(self, stats: Dict[str, float]) -> float:
        """
        現在の状態に対する報酬を計算
        
        Parameters:
        -----------
        stats: 統計情報
        
        Returns:
        --------
        報酬値
        """
        # エントロピー誤差に基づく報酬
        entropy_err = abs(stats.get('entropy', 0) - self.target_H)
        entropy_reward = 1.0 / (1.0 + entropy_err)
        
        # VDIに基づく報酬
        vdi_reward = stats.get('vdi', 0)
        
        # FCRに基づく報酬
        fcr_reward = stats.get('fcr', 0)
        
        # 速度に基づく報酬
        speed = stats.get('speed', 0)
        base_speed = await self.bb.get_param('base_speed', 10.0)
        speed_ratio = speed / base_speed if base_speed > 0 else 0
        speed_reward = min(speed_ratio, 1.0)
        
        # 総合報酬
        reward = 0.4 * entropy_reward + 0.2 * vdi_reward + 0.2 * fcr_reward + 0.2 * speed_reward        
        return reward
    
    async def run(self, interval: float = 1.0):
        """
        コントローラーを非同期で実行
        
        Parameters:
        -----------
        interval: 調整間隔 (秒)
        """
        try:
            while not self.early_stop:
                # シャットダウンフラグをチェック
                import sys
                run_sim_module = sys.modules.get('slm_emergent_ai.run_sim') or sys.modules.get('__main__')
                global_shutdown = getattr(run_sim_module, 'shutdown_requested', False) if run_sim_module else False
                
                if global_shutdown:
                    print("[INFO] MetaController received global shutdown signal, stopping.")
                    self.early_stop = True
                    break
                
                # 統計情報を取得
                stats = {
                    'entropy': await self.bb.get_param('entropy', 0.0),
                    'vdi': await self.bb.get_param('vdi', 0.0),
                    'fcr': await self.bb.get_param('fcr', 0.0),
                    'speed': await self.bb.get_param('speed', 0.0)
                }
                
                # 現在の状態を取得
                state = await self._discretize_state(stats)
                
                # 行動を選択
                action = self._select_action(state)
                
                # λパラメータを調整
                await self.adjust_lambda(stats)
                
                # 調整後の統計情報を取得
                await asyncio.sleep(interval)
                
                # 再度シャットダウンチェック
                if self.early_stop or global_shutdown:
                    break
                    
                new_stats = {
                    'entropy': await self.bb.get_param('entropy', 0.0),
                    'vdi': await self.bb.get_param('vdi', 0.0),
                    'fcr': await self.bb.get_param('fcr', 0.0),
                    'speed': await self.bb.get_param('speed', 0.0)
                }
                
                # 次の状態を取得
                next_state = await self._discretize_state(new_stats)
                
                # 報酬を計算
                reward = await self.calculate_reward(new_stats)
                
                # Q値を更新
                self.update_Q(state, action, reward, next_state)
        except asyncio.CancelledError:
            print("[INFO] MetaController run loop cancelled gracefully.")
        except Exception as e:
            print(f"[ERROR] MetaController run loop failed: {e}")
        finally:
            print("[INFO] MetaController shut down complete.")
