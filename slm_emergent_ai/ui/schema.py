"""
UI Schema - FastAPI用のスキーマ定義

UIダッシュボードで使用するAPIスキーマの定義。
Pydanticモデルを使用してリクエスト/レスポンスの型を定義する。
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field


class MetricsData(BaseModel):
    """メトリクスデータのスキーマ"""
    entropy: float = Field(0.0, description="エントロピー値")
    vdi: float = Field(0.0, description="語彙多様性指数")
    fcr: float = Field(0.0, description="Fake Correction Rate")
    speed: float = Field(0.0, description="生成速度 (tok/s)")
    lambda_a: float = Field(0.3, description="λ_a パラメータ")
    lambda_c: float = Field(0.3, description="λ_c パラメータ")
    lambda_s: float = Field(0.1, description="λ_s パラメータ")
    token_count: int = Field(0, description="生成されたトークン数")
    elapsed_time: float = Field(0.0, description="経過時間")


class SystemStatus(BaseModel):
    """システム状態のスキーマ"""
    ram: float = Field(0.0, description="RAM使用率 (%)")
    cpu: float = Field(0.0, description="CPU使用率 (%)")
    swap: float = Field(0.0, description="SWAP使用率 (%)")
    disk: float = Field(0.0, description="ディスク使用率 (%)")


class BlackboardMessage(BaseModel):
    """BlackBoardメッセージのスキーマ"""
    agent_id: int = Field(..., description="エージェントID")
    text: str = Field(..., description="メッセージテキスト")
    timestamp: Optional[float] = Field(None, description="タイムスタンプ")


class PromptInjectRequest(BaseModel):
    """プロンプト注入リクエストのスキーマ"""
    prompt: str = Field(..., description="注入するプロンプト")
    agent_id: Optional[int] = Field(0, description="対象エージェントID (0は全エージェント)")


class PromptInjectResponse(BaseModel):
    """プロンプト注入レスポンスのスキーマ"""
    success: bool = Field(..., description="成功したかどうか")
    error: Optional[str] = Field(None, description="エラーメッセージ")


class AgentConfig(BaseModel):
    """エージェント設定のスキーマ"""
    id: int = Field(..., description="エージェントID")
    role: str = Field(..., description="エージェントの役割")
    lambda_a: float = Field(0.3, description="λ_a パラメータ")
    lambda_c: float = Field(0.3, description="λ_c パラメータ")
    lambda_s: float = Field(0.1, description="λ_s パラメータ")
    enabled: bool = Field(True, description="有効かどうか")


class AgentConfigUpdateRequest(BaseModel):
    """エージェント設定更新リクエストのスキーマ"""
    agents: List[AgentConfig] = Field(..., description="エージェント設定のリスト")


class AgentConfigUpdateResponse(BaseModel):
    """エージェント設定更新レスポンスのスキーマ"""
    success: bool = Field(..., description="成功したかどうか")
    error: Optional[str] = Field(None, description="エラーメッセージ")


class ControllerConfig(BaseModel):
    """コントローラー設定のスキーマ"""
    target_H: float = Field(5.0, description="目標エントロピー")
    Kp: float = Field(0.1, description="比例ゲイン")
    Ki: float = Field(0.01, description="積分ゲイン")
    Kd: float = Field(0.05, description="微分ゲイン")
    early_stop: bool = Field(False, description="EarlyStop有効化")


class ControllerConfigUpdateRequest(BaseModel):
    """コントローラー設定更新リクエストのスキーマ"""
    config: ControllerConfig = Field(..., description="コントローラー設定")


class ControllerConfigUpdateResponse(BaseModel):
    """コントローラー設定更新レスポンスのスキーマ"""
    success: bool = Field(..., description="成功したかどうか")
    error: Optional[str] = Field(None, description="エラーメッセージ")


class WebSocketMessage(BaseModel):
    """WebSocketメッセージのスキーマ"""
    type: str = Field(..., description="メッセージタイプ")
    data: Dict[str, Any] = Field(..., description="メッセージデータ")


class WebSocketSubscribeRequest(BaseModel):
    """WebSocketサブスクリプションリクエストのスキーマ"""
    type: str = Field("subscribe", description="メッセージタイプ")
    channel: str = Field(..., description="サブスクライブするチャンネル")


class SimulationControlRequest(BaseModel):
    """シミュレーション制御リクエストのスキーマ"""
    action: str = Field(..., description="アクション (start, stop, pause, resume)")
    config: Optional[Dict[str, Any]] = Field(None, description="設定 (startアクション時のみ)")


class SimulationControlResponse(BaseModel):
    """シミュレーション制御レスポンスのスキーマ"""
    success: bool = Field(..., description="成功したかどうか")
    status: str = Field(..., description="現在のステータス")
    error: Optional[str] = Field(None, description="エラーメッセージ")


class ExportRequest(BaseModel):
    """データエクスポートリクエストのスキーマ"""
    type: str = Field(..., description="エクスポートタイプ (metrics, blackboard, all)")
    format: str = Field("json", description="エクスポート形式 (json, csv)")


class ExportResponse(BaseModel):
    """データエクスポートレスポンスのスキーマ"""
    success: bool = Field(..., description="成功したかどうか")
    file_path: Optional[str] = Field(None, description="エクスポートされたファイルパス")
    error: Optional[str] = Field(None, description="エラーメッセージ")


class BoidsConfig(BaseModel):
    """BOIDSルール設定のスキーマ"""
    lambda_a: float = Field(0.3, description="Alignment weight (共感性)")
    lambda_c: float = Field(0.3, description="Cohesion weight (結束力)")
    lambda_s: float = Field(0.1, description="Separation weight (独自性)")
    target_entropy: float = Field(5.0, description="目標エントロピー値")
    similarity_threshold: float = Field(0.9, description="類似度閾値")
    diversity_threshold: float = Field(0.7, description="多様性閾値")
    alignment_threshold: float = Field(0.8, description="共感閾値")
    cohesion_threshold: float = Field(0.6, description="結束閾値")
    enabled: bool = Field(True, description="BOIDSルール有効化")


class BoidsConfigUpdateRequest(BaseModel):
    """BOIDSルール設定更新リクエストのスキーマ"""
    config: BoidsConfig = Field(..., description="BOIDSルール設定")


class BoidsConfigUpdateResponse(BaseModel):
    """BOIDSルール設定更新レスポンスのスキーマ"""
    success: bool = Field(..., description="成功したかどうか")
    updated_config: Optional[BoidsConfig] = Field(None, description="更新後の設定")
    error: Optional[str] = Field(None, description="エラーメッセージ")


class RealTimeMetrics(BaseModel):
    """リアルタイムメトリクスのスキーマ"""
    entropy: float = Field(0.0, description="現在のエントロピー値")
    vdi: float = Field(0.0, description="語彙多様性指数")
    fcr: float = Field(0.0, description="議論の一貫性")
    speed: float = Field(0.0, description="会話のテンポ")
    boids_weights: BoidsConfig = Field(..., description="現在のBOIDSパラメータ")
    adjustment_history: List[Dict[str, float]] = Field([], description="調整履歴")
