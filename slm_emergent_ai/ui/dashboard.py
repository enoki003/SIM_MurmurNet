"""
UI Dashboard - FastAPI + Vue3 SPAのダッシュボード

SLM Emergent AIのUIダッシュボード。
WebSocketでmetrics/bb/streamを購読し、リアルタイムでグラフを更新。
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional

# Import BlackBoard if it exists in another module
try:
    from ..memory.blackboard import BlackBoard  # Corrected import
except ImportError:
    # Fallback BlackBoard definition for standalone UI testing or if main module is not in PYTHONPATH
    print("[Dashboard WARNING] Actual BlackBoard class not found, using placeholder for UI.")
    class BlackBoard:
        """Placeholder BlackBoard class for UI when the main module is not accessible."""
        async def get_param(self, key: str, default: Any) -> Any: return default
        async def pull(self, k: int) -> List[Dict[str, Any]]: return []
        async def push(self, message: Dict[str, Any]) -> None: pass
        async def pull_messages_raw(self, k: int) -> List[Dict[str, Any]]: return []

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

class Dashboard:
    """SLM Emergent AIのダッシュボード"""
    def __init__(self, port: int = 7860, bb: Optional['BlackBoard'] = None, metrics_server_url: str = "http://localhost:7861"):
        """
        Dashboardの初期化
        
        Parameters:
        -----------
        port: サーバーポート
        bb: BlackBoardインスタンス
        metrics_server_url: メトリクスサーバーのURL
        """
        self.port = port
        self.bb = bb
        self.metrics_server_url = metrics_server_url
        self.app = None
        self.clients = []
        self._initialize()
    
    def _initialize(self):
        """FastAPIアプリの初期化"""
        try:
            self.app = FastAPI(title="SLM Emergent AI Dashboard")
            
            # CORS設定
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            # 静的ファイルの設定
            static_dir = os.path.join(os.path.dirname(__file__), "static")
            os.makedirs(static_dir, exist_ok=True)
            self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
            
            # HTMLレスポンスの設定
            @self.app.get("/", response_class=HTMLResponse)
            async def get_dashboard():
                return self._get_dashboard_html()
            
            # WebSocketエンドポイント
            @self.app.websocket("/ws")
            async def websocket_endpoint(websocket: WebSocket):
                await websocket.accept()
                self.clients.append(websocket)
                try:
                    while True:
                        data = await websocket.receive_text()
                        await self._handle_websocket_message(websocket, data)
                except WebSocketDisconnect:
                    self.clients.remove(websocket)
            
            # APIエンドポイント
            @self.app.get("/api/metrics")
            async def get_metrics():
                return await self._fetch_metrics()
            
            @self.app.get("/api/blackboard")
            async def get_blackboard():
                return await self._fetch_blackboard()
            
            @self.app.post("/api/inject")
            async def inject_prompt(data: Dict[str, str]):
                return await self._inject_prompt(data.get("prompt", ""))
            
            # BOIDSパラメータ調整エンドポイント
            @self.app.get("/api/boids/config")
            async def get_boids_config():
                return await self._get_boids_config()
            
            @self.app.post("/api/boids/config")
            async def update_boids_config(request: Dict[str, Any]):
                return await self._update_boids_config(request)
            
            @self.app.get("/api/boids/presets")
            async def get_boids_presets():
                return await self._get_boids_presets()
            
            @self.app.post("/api/boids/presets/{preset_name}")
            async def apply_boids_preset(preset_name: str):
                return await self._apply_boids_preset(preset_name)
            
            print(f"Dashboard initialized on port {self.port}")
        except Exception as e:
            print(f"Error initializing dashboard: {e}")
    
    async def _handle_websocket_message(self, websocket: WebSocket, data: str):
        """WebSocketメッセージを処理"""
        try:
            message = json.loads(data)
            if message.get("type") == "subscribe":
                channel = message.get("channel")
                print(f"Client subscribed to channel: {channel}")
            elif message.get("type") == "metrics": # Assuming metrics are pushed via WebSocket
                print(f"[METRICS_DEBUG] Dashboard WebSocket received metrics: {message.get('data')}")
                # This part would then update the Vue app's metrics ref,
                # which is already handled by the Vue code if it receives 'metrics' type messages.
            elif message.get("type") == "inject":
                prompt = message.get("prompt")
                await self._inject_prompt(prompt)
        except Exception as e:
            print(f"Error handling WebSocket message: {e}")
    
    async def _fetch_metrics(self) -> Dict[str, Any]:
        """メトリクスを取得"""
        metrics_to_return = {}
        try:
            if self.bb:
                try:
                    entropy = await self.bb.get_param("entropy", 0.0)
                    target_H = await self.bb.get_param("target_H", 5.0)
                    
                    # BOIDSパラメータを取得
                    lambda_c = await self.bb.get_param("λ_c", 0.3)
                    lambda_a = await self.bb.get_param("λ_a", 0.3)
                    lambda_s = await self.bb.get_param("λ_s", 0.1)
                    boids_enabled = await self.bb.get_param("boids_enabled", True)
                    
                    # 自動調整状況の情報を追加
                    vdi = await self.bb.get_param("vdi", 0.0)
                    auto_adjustment_status = "正常"
                    
                    # VDIベースの自動調整状況を判定
                    if vdi < 0.3:
                        auto_adjustment_status = "多様性不足→独自性強化中"
                    elif vdi > 0.7:
                        auto_adjustment_status = "多様性過多→収束促進中"
                    elif abs(entropy - target_H) > 1.0:
                        auto_adjustment_status = "エントロピー調整中"
                    
                    metrics_to_return = {
                        "entropy": float(entropy),
                        "vdi": float(vdi),
                        "fcr": float(await self.bb.get_param("fcr", 1.0)),
                        "speed": float(await self.bb.get_param("speed", 0.0)),
                        "target_H": float(target_H),
                        "lambda_c": float(lambda_c),
                        "lambda_a": float(lambda_a),
                        "lambda_s": float(lambda_s),
                        "boids_enabled": bool(boids_enabled),
                        "auto_adjustment_status": auto_adjustment_status,
                        "stability_check": lambda_a + lambda_c > lambda_s,
                        "entropy_error": entropy - target_H
                    }
                    
                    print(f"[BOIDS_DEBUG] Dashboard._fetch_metrics BOIDS params: λ_c={lambda_c:.3f}, λ_a={lambda_a:.3f}, λ_s={lambda_s:.3f}, enabled={boids_enabled}")
                    print(f"[METRICS_DEBUG] Dashboard._fetch_metrics returning from bb.get_param: {metrics_to_return}")
                    return metrics_to_return
                except Exception as e_get_param:
                    print(f"[METRICS_DEBUG] Error fetching individual params from BlackBoard: {e_get_param}")
                    pass
                
                # 代替手段としてpullメソッドを使用してメトリクスを取得
                try:
                    metrics_list = await self.bb.pull(k=10)  # 最新のメトリクスを10件取得
                    print(f"[METRICS_DEBUG] Fetched metrics from pull: {metrics_list}")
                    
                    # メトリクスリストから最新の値を抽出
                    for metric in metrics_list:
                        if isinstance(metric, dict):
                            for key, value in metric.items():
                                if key in ["entropy", "vdi", "fcr", "speed", "target_H"]:
                                    metrics_to_return[key] = float(value)
                
                except Exception as e_pull:
                    print(f"[METRICS_DEBUG] Error fetching metrics from pull: {e_pull}")
                    raise e_pull  # Reraise to be caught by the outer exception handler
                
                # If metrics_to_return is still empty, use a default structure.
                if not metrics_to_return:
                    metrics_to_return = {
                        "entropy": 0.0, "vdi": 0.0, "fcr": 1.0, "speed": 0.0, "target_H": 0.0
                    }
                print(f"[METRICS_DEBUG] Dashboard._fetch_metrics (after potential fallback): {metrics_to_return}")
                return metrics_to_return
            else:
                print("[METRICS_DEBUG] Dashboard._fetch_metrics: BlackBoard not available.")
                # Return a default structure if BlackBoard is not available
                return { "entropy": 0.0, "vdi": 0.0, "fcr": 1.0, "speed": 0.0, "target_H": 0.0 }
        except Exception as e:
            print(f"[METRICS_DEBUG] Error in _fetch_metrics: {e}")
            return { "entropy": 0.0, "vdi": 0.0, "fcr": 1.0, "speed": 0.0, "target_H": 0.0 }

    async def _get_boids_config(self) -> Dict[str, Any]:
        """現在のBOIDSパラメータ設定を取得"""
        try:
            if self.bb:
                config = {
                    "lambda_a": await self.bb.get_param("λ_a", 0.3),
                    "lambda_c": await self.bb.get_param("λ_c", 0.3),
                    "lambda_s": await self.bb.get_param("λ_s", 0.1),
                    "target_entropy": await self.bb.get_param("target_H", 5.0),
                    "enabled": await self.bb.get_param("boids_enabled", True),
                    "similarity_threshold": await self.bb.get_param("similarity_threshold", 0.9),
                    "diversity_threshold": await self.bb.get_param("diversity_threshold", 0.7),
                    "alignment_threshold": await self.bb.get_param("alignment_threshold", 0.8),
                    "cohesion_threshold": await self.bb.get_param("cohesion_threshold", 0.6)
                }
                return {"success": True, "config": config}
            else:
                return {"success": False, "error": "BlackBoard not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _update_boids_config(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """BOIDSパラメータ設定を更新"""
        try:
            if not self.bb:
                return {"success": False, "error": "BlackBoard not available"}
            
            config = request.get("config", {})
            
            # パラメータの検証と制約
            lambda_a = max(0.0, min(1.0, config.get("lambda_a", 0.3)))
            lambda_c = max(0.0, min(1.0, config.get("lambda_c", 0.3)))
            lambda_s = max(0.0, min(1.0, config.get("lambda_s", 0.1)))
            
            # 安定性制約の確認
            if lambda_a + lambda_c <= lambda_s:
                lambda_s = max(0.0, lambda_a + lambda_c - 0.05)
            
            target_entropy = max(0.0, min(10.0, config.get("target_entropy", 5.0)))
            
            # BlackBoardに設定を保存
            await self.bb.set_param("λ_a", lambda_a)
            await self.bb.set_param("λ_c", lambda_c)
            await self.bb.set_param("λ_s", lambda_s)
            await self.bb.set_param("target_H", target_entropy)
            await self.bb.set_param("boids_enabled", config.get("enabled", True))
            
            # 閾値パラメータも更新
            await self.bb.set_param("similarity_threshold", config.get("similarity_threshold", 0.9))
            await self.bb.set_param("diversity_threshold", config.get("diversity_threshold", 0.7))
            await self.bb.set_param("alignment_threshold", config.get("alignment_threshold", 0.8))
            await self.bb.set_param("cohesion_threshold", config.get("cohesion_threshold", 0.6))
            
            # 更新通知をブロードキャスト
            await self.broadcast({
                "type": "boids_config_updated",
                "data": {
                    "lambda_a": lambda_a,
                    "lambda_c": lambda_c,
                    "lambda_s": lambda_s,
                    "target_entropy": target_entropy,
                    "enabled": config.get("enabled", True)
                }
            })
            
            updated_config = {
                "lambda_a": lambda_a,
                "lambda_c": lambda_c,
                "lambda_s": lambda_s,
                "target_entropy": target_entropy,
                "enabled": config.get("enabled", True)
            }
            
            return {"success": True, "updated_config": updated_config}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _get_boids_presets(self) -> Dict[str, Any]:
        """BOIDSプリセット設定を取得"""
        presets = {
            "balanced": {
                "name": "バランス重視",
                "description": "議論のバランスを重視した設定",
                "lambda_a": 0.3,
                "lambda_c": 0.3,
                "lambda_s": 0.1,
                "target_entropy": 5.0
            },
            "creative": {
                "name": "創造性重視",
                "description": "創造的で多様な発言を促進",
                "lambda_a": 0.2,
                "lambda_c": 0.2,
                "lambda_s": 0.3,
                "target_entropy": 6.5
            },
            "focused": {
                "name": "集中重視",
                "description": "話題の集中と一貫性を重視",
                "lambda_a": 0.4,
                "lambda_c": 0.5,
                "lambda_s": 0.05,
                "target_entropy": 3.5
            },
            "dynamic": {
                "name": "動的議論",
                "description": "活発で動的な議論を促進",
                "lambda_a": 0.35,
                "lambda_c": 0.25,
                "lambda_s": 0.2,
                "target_entropy": 5.5
            }
        }
        return {"success": True, "presets": presets}
    
    async def _apply_boids_preset(self, preset_name: str) -> Dict[str, Any]:
        """BOIDSプリセットを適用"""
        try:
            presets_response = await self._get_boids_presets()
            presets = presets_response.get("presets", {})
            
            if preset_name not in presets:
                return {"success": False, "error": f"Unknown preset: {preset_name}"}
            
            preset = presets[preset_name]
            config_request = {"config": preset}
            
            return await self._update_boids_config(config_request)
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _fetch_blackboard(self) -> Dict[str, Any]:
        """BlackBoardの情報を取得"""
        try:
            if self.bb:
                # BlackBoardから最新のメッセージを辞書形式で取得
                messages = await self.bb.pull_messages_raw(k=50)  # 最新50件を取得
                
                # メッセージを整形
                formatted_messages = []
                for i, msg in enumerate(messages):
                    if isinstance(msg, dict):
                        # メッセージが辞書形式の場合
                        agent_id = msg.get('agent_id', f'Agent_{i}')
                        role = msg.get('role', f'Agent_{agent_id}')
                        text = msg.get('text', '')
                        
                        # テキストが空または単なるトークン番号の場合はスキップ
                        if not text or text.startswith('token_') and len(text) < 20:
                            continue
                            
                        # トークン番号だけの場合は除外
                        if len(text.strip()) < 3:
                            continue

                        # エージェント表示名のフォーマット
                        display_name = role if role != f'Agent_{agent_id}' else f'Agent {agent_id}'
                        
                        formatted_messages.append({
                            'agent_id': display_name,
                            'text': text,
                            'timestamp': msg.get('timestamp', '')
                        })
                    else:
                        # メッセージが文字列の場合
                        if isinstance(msg, str) and len(msg.strip()) > 3:
                            # トークン番号だけの文字列は除外
                            if msg.startswith('token_') and len(msg) < 20:
                                continue
                            
                            formatted_messages.append({
                                'agent_id': f'Agent {i % 5 + 1}',  # 5つのエージェントIDを循環
                                'text': str(msg),
                                'timestamp': ''
                            })
                
                # メッセージがない場合はダミーメッセージを追加
                if not formatted_messages:
                    formatted_messages.append({
                        'agent_id': 'System',
                        'text': 'エージェントの会話がまだ開始されていません。',
                        'timestamp': __import__('time').strftime("%H:%M:%S")
                    })
                # 最新のメッセージを先頭に
                formatted_messages.reverse()
                
                return {"messages": formatted_messages}
            else:
                # BlackBoardが利用できない場合
                print("BlackBoard not available for dashboard")
                return {"messages": []}
        except Exception as e:
            print(f"Error fetching blackboard: {e}")
            return {"messages": []}
    
    async def _inject_prompt(self, prompt: str) -> Dict[str, Any]:
        """プロンプトを注入"""
        try:
            if not prompt.strip():
                return {"status": "error", "message": "Empty prompt"}
            
            print(f"Injecting prompt: {prompt}")
            
            if self.bb:
                # BlackBoardにプロンプトを注入
                injection_message = {
                    "type": "user_injection",
                    "text": prompt,
                    "agent_id": "USER",
                    "timestamp": __import__('time').strftime("%H:%M:%S"),
                    "priority": "high"
                }
                
                # BlackBoardにメッセージを送信
                await self.bb.push(injection_message)
                
                # 全クライアントに注入されたプロンプトを通知
                await self.broadcast({
                    "type": "prompt_injected",
                    "data": {
                        "prompt": prompt,
                        "status": "success",
                        "timestamp": injection_message["timestamp"]
                    }
                })
                
                return {"status": "success", "message": "Prompt injected successfully"}
            else:
                print(f"BlackBoard not available, prompt logged: {prompt}")
                return {"status": "success", "message": "Prompt logged (BlackBoard not available)"}
            
        except Exception as e:
            print(f"Error injecting prompt: {e}")
            return {"status": "error", "message": str(e)}
    
    def _get_dashboard_html(self) -> str:
        """ダッシュボードのHTMLを取得"""
        return """
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SLM Emergent AI Dashboard</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/vue@3.2.47/dist/vue.global.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
            <style>
                body { padding: 20px; background-color: #f8f9fa; }
                .card { margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                .card-header { background-color: #343a40; color: white; }
                .metrics-value { font-size: 24px; font-weight: bold; }
                .chart-container { height: 300px; }
                .log-container { height: 300px; overflow-y: auto; background-color: #f5f5f5; padding: 10px; border-radius: 5px; }
                .log-entry { margin-bottom: 5px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
                .control-panel { background-color: #e9ecef; padding: 15px; border-radius: 5px; }
                .conversation-text { font-style: italic; color: #495057; line-height: 1.4; }
                .boids-control { background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107; }
                .preset-button { margin: 2px; }
                .slider-container { margin: 10px 0; }
                .slider-label { display: flex; justify-content: space-between; font-size: 0.9em; margin-bottom: 5px; }
                .range-input { width: 100%; }
            </style>
        </head>
        <body>
            <div id="app" class="container-fluid">
                <h1 class="mb-4">SLM Emergent AI Dashboard</h1>
                
                <div class="row">
                    <div class="col-md-8">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">Metrics Over Time</h5>
                            </div>
                            <div class="card-body">
                                <div class="chart-container">
                                    <canvas id="metricsChart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">Current Metrics</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">エントロピー (H)</div>
                                        <div class="metrics-value">{{ typeof metrics.entropy === 'number' ? metrics.entropy.toFixed(2) : '0.00' }}</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">多様性 (VDI)</div>
                                        <div class="metrics-value">{{ typeof metrics.vdi === 'number' ? metrics.vdi.toFixed(2) : '0.00' }}</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">協調性 (FCR)</div>
                                        <div class="metrics-value">{{ typeof metrics.fcr === 'number' ? metrics.fcr.toFixed(2) : '0.00' }}</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">速度 (トークン/秒)</div>
                                        <div class="metrics-value">{{ typeof metrics.speed === 'number' ? metrics.speed.toFixed(2) : '0.00' }}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- 新しいBOIDSデバッグセクション -->
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">BOIDS Rule Debug</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">λ_c (Cohesion)</div>
                                        <div class="metrics-value">{{ typeof metrics.lambda_c === 'number' ? metrics.lambda_c.toFixed(3) : '0.000' }}</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">λ_a (Alignment)</div>
                                        <div class="metrics-value">{{ typeof metrics.lambda_a === 'number' ? metrics.lambda_a.toFixed(3) : '0.000' }}</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">λ_s (Separation)</div>
                                        <div class="metrics-value">{{ typeof metrics.lambda_s === 'number' ? metrics.lambda_s.toFixed(3) : '0.000' }}</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">Target H</div>
                                        <div class="metrics-value">{{ typeof metrics.target_H === 'number' ? metrics.target_H.toFixed(2) : '0.00' }}</div>
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-12 mb-2">
                                        <div class="text-muted">BOIDS Rule Status</div>
                                        <div class="small text-success" v-if="metrics.boids_enabled">✓ 有効</div>
                                        <div class="small text-warning" v-else>✗ 無効</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-8">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">BlackBoard Messages</h5>
                            </div>
                            <div class="card-body">
                                <div class="log-container">
                                    <div v-for="(message, index) in blackboardMessages" :key="index" class="log-entry">
                                        <div class="d-flex">
                                            <span class="me-2 text-primary"><strong>{{ message.agent_id }}:</strong></span>
                                            <span>{{ message.text }}</span>
                                            <small class="ms-2 text-muted">{{ message.timestamp }}</small>
                                        </div>
                                    </div>
                                    <div v-if="blackboardMessages.length === 0" class="text-center text-muted">
                                        メッセージはまだありません
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-4">
                        <!-- BOIDSパラメータ調整パネル -->
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">会話ダイナミクス調整</h5>
                            </div>
                            <div class="card-body">
                                <div class="boids-control">
                                    <!-- プリセットボタン -->
                                    <div class="mb-3">
                                        <label class="form-label"><strong>プリセット設定</strong></label>
                                        <div>
                                            <button v-for="(preset, key) in boidsPresets" :key="key" 
                                                    class="btn btn-sm btn-outline-primary preset-button"
                                                    @click="applyPreset(key)">
                                                {{ preset.name }}
                                            </button>
                                        </div>
                                    </div>
                                    
                                    <!-- 手動調整スライダー -->
                                    <div class="slider-container">
                                        <div class="slider-label">
                                            <span>共感性 (λ_a)</span>
                                            <span>{{ boidsConfig.lambda_a.toFixed(3) }}</span>
                                        </div>
                                        <input type="range" class="range-input" 
                                               v-model="boidsConfig.lambda_a" 
                                               min="0" max="0.8" step="0.01"
                                               @input="updateBoidsConfig">
                                    </div>
                                    
                                    <div class="slider-container">
                                        <div class="slider-label">
                                            <span>結束力 (λ_c)</span>
                                            <span>{{ boidsConfig.lambda_c.toFixed(3) }}</span>
                                        </div>
                                        <input type="range" class="range-input" 
                                               v-model="boidsConfig.lambda_c" 
                                               min="0" max="0.8" step="0.01"
                                               @input="updateBoidsConfig">
                                    </div>
                                    
                                    <div class="slider-container">
                                        <div class="slider-label">
                                            <span>独自性 (λ_s)</span>
                                            <span>{{ boidsConfig.lambda_s.toFixed(3) }}</span>
                                        </div>
                                        <input type="range" class="range-input" 
                                               v-model="boidsConfig.lambda_s" 
                                               min="0" max="0.5" step="0.01"
                                               @input="updateBoidsConfig">
                                    </div>
                                    
                                    <div class="slider-container">
                                        <div class="slider-label">
                                            <span>目標多様性</span>
                                            <span>{{ boidsConfig.target_entropy.toFixed(1) }}</span>
                                        </div>
                                        <input type="range" class="range-input" 
                                               v-model="boidsConfig.target_entropy" 
                                               min="1" max="10" step="0.1"
                                               @input="updateBoidsConfig">
                                    </div>
                                    
                                    <div class="form-check mt-3">
                                        <input class="form-check-input" type="checkbox" 
                                               v-model="boidsConfig.enabled" 
                                               @change="updateBoidsConfig" id="boidsEnabled">
                                        <label class="form-check-label" for="boidsEnabled">
                                            BOIDSルール有効化
                                        </label>
                                    </div>
                                    
                                    <div class="mt-3">
                                        <small class="text-muted">
                                            安定性制約: λ_a + λ_c > λ_s
                                            <br>
                                            現在: {{ (parseFloat(boidsConfig.lambda_a) + parseFloat(boidsConfig.lambda_c)).toFixed(3) }} > {{ parseFloat(boidsConfig.lambda_s).toFixed(3) }}
                                            <span v-if="parseFloat(boidsConfig.lambda_a) + parseFloat(boidsConfig.lambda_c) > parseFloat(boidsConfig.lambda_s)" class="text-success">✓</span>
                                            <span v-else class="text-danger">✗</span>
                                        </small>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">Control Panel</h5>
                            </div>
                            <div class="card-body">
                                <div class="control-panel">
                                    <div class="mb-3">
                                        <label for="promptInput" class="form-label">Live Prompt Inject</label>
                                        <textarea class="form-control" id="promptInput" v-model="promptInput" rows="3" placeholder="Enter prompt to inject..."></textarea>
                                    </div>
                                    <button class="btn btn-primary" @click="injectPrompt">Inject Prompt</button>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">System Status</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">RAM Usage</div>
                                        <div class="metrics-value">{{ systemStatus.ram }}%</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">CPU Usage</div>
                                        <div class="metrics-value">{{ systemStatus.cpu }}%</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                const { createApp, ref, onMounted, onUnmounted } = Vue;
                
                createApp({
                    setup() {
                        // リアクティブデータ
                        const metrics = ref({
                            entropy: 4.5,
                            vdi: 0.7,
                            fcr: 0.9,
                            speed: 15.0
                        });
                        
                        const blackboardMessages = ref([]);
                        const promptInput = ref('');
                        const systemStatus = ref({
                            ram: 0,
                            cpu: 0
                        });
                        
                        const boidsConfig = ref({
                            lambda_a: 0.3,
                            lambda_c: 0.3,
                            lambda_s: 0.1,
                            target_entropy: 5.0,
                            enabled: true
                        });
                        
                        const boidsPresets = ref({});
                        
                        // WebSocket接続
                        let socket = null;
                        
                        onMounted(() => {
                            // ソケット接続
                            socket = new WebSocket(`ws://${window.location.host}/ws`);
                            
                            socket.onopen = () => {
                                console.log("WebSocket connected");
                            };
                            
                            socket.onmessage = (event) => {
                                const message = JSON.parse(event.data);
                                handleSocketMessage(message);
                            };
                            
                            socket.onclose = () => {
                                console.log("WebSocket disconnected, retrying in 1 second...");
                                setTimeout(() => {
                                    connectWebSocket();
                                }, 1000);
                            };
                        });
                        
                        onUnmounted(() => {
                            if (socket) {
                                socket.close();
                            }
                        });
                        
                        function handleSocketMessage(message) {
                            if (message.type === "metrics") {
                                metrics.value = message.data;
                            } else if (message.type === "blackboard") {
                                // BlackBoardメッセージの処理
                                blackboardMessages.value = message.data.messages;
                            } else if (message.type === "prompt_injected") {
                                // プロンプト注入メッセージの処理
                                const injectedMessage = {
                                    agent_id: "あなた",
                                    text: message.data.prompt,
                                    timestamp: message.data.timestamp
                                };
                                blackboardMessages.value.unshift(injectedMessage);
                            }
                        }
                        
                        async function fetchInitialData() {
                            try {
                                // 初期メトリクスの取得
                                const metricsResponse = await fetch("/api/metrics");
                                if (metricsResponse.ok) {
                                    metrics.value = await metricsResponse.json();
                                }
                                
                                // 初期BlackBoardメッセージの取得
                                const blackboardResponse = await fetch("/api/blackboard");
                                if (blackboardResponse.ok) {
                                    const blackboardData = await blackboardResponse.json();
                                    blackboardMessages.value = blackboardData.messages;
                                }
                                
                                // BOIDSプリセットの取得
                                const presetsResponse = await fetch("/api/boids/presets");
                                if (presetsResponse.ok) {
                                    const presetsData = await presetsResponse.json();
                                    boidsPresets.value = presetsData.presets;
                                }
                            } catch (error) {
                                console.error("Error fetching initial data:", error);
                            }
                        }
                        
                        function connectWebSocket() {
                            socket = new WebSocket(`ws://${window.location.host}/ws`);
                            
                            socket.onopen = () => {
                                console.log("WebSocket re-connected");
                            };
                            
                            socket.onmessage = (event) => {
                                const message = JSON.parse(event.data);
                                handleSocketMessage(message);
                            };
                        }
                        
                        // プロンプト注入
                        async function injectPrompt() {
                            if (!promptInput.value.trim()) return;
                            
                            try {
                                const response = await fetch("/api/inject", {
                                    method: "POST",
                                    headers: {
                                        "Content-Type": "application/json"
                                    },
                                    body: JSON.stringify({ prompt: promptInput.value })
                                });
                                
                                const result = await response.json();
                                if (result.status === "success") {
                                    promptInput.value = "";
                                } else {
                                    console.error("Error injecting prompt:", result.message);
                                }
                            } catch (error) {
                                console.error("Error injecting prompt:", error);
                            }
                        }
                        
                        // BOIDS設定の手動調整
                        async function updateBoidsConfig() {
                            try {
                                await fetch("/api/boids/config", {
                                    method: "POST",
                                    headers: {
                                        "Content-Type": "application/json"
                                    },
                                    body: JSON.stringify({ config: boidsConfig.value })
                                });
                            } catch (error) {
                                console.error("Error updating BOIDS config:", error);
                            }
                        }
                        
                        return {
                            metrics,
                            blackboardMessages,
                            promptInput,
                            systemStatus,
                            boidsConfig,
                            boidsPresets,
                            injectPrompt,
                            updateBoidsConfig
                        };
                    }
                }).mount("#app");
            </script>
        </body>
        </html>
        """
    
    async def broadcast(self, message: Dict[str, Any]):
        """全クライアントにメッセージをブロードキャスト"""
        for client in self.clients:
            try:
                await client.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                self.clients.remove(client)
    
    def start(self):
        """サーバーを起動"""
        if self.app:
            import uvicorn
            uvicorn.run(self.app, host="0.0.0.0", port=self.port)
        else:
            print("Error: App not initialized")
    
    async def run_metrics_updater(self, interval: float = 1.0):
        """メトリクス更新ループを実行"""
        while True:
            try:
                metrics = await self._fetch_metrics()
                await self.broadcast({"type": "metrics", "data": metrics})
                await asyncio.sleep(interval)
            except Exception as e:
                print(f"Error in metrics updater: {e}")
                await asyncio.sleep(interval)
    
    async def run_blackboard_updater(self, interval: float = 0.5):
        """BlackBoard更新ループを実行"""
        while True:
            try:
                blackboard = await self._fetch_blackboard()
                await self.broadcast({"type": "blackboard", "data": blackboard})
                await asyncio.sleep(interval)
            except Exception as e:
                print(f"Error in blackboard updater: {e}")
                await asyncio.sleep(interval)


def init_app(blackboard=None, port: int = 7860, metrics_server_url: str = "http://localhost:7861"):
    """
    FastAPIアプリを初期化して返す関数
    
    Parameters:
    -----------
    blackboard: BlackBoardインスタンス
    port: サーバーポート (使用されないが互換性のため保持)
    metrics_server_url: メトリクスサーバーのURL
    
    Returns:
    --------
    FastAPI: 初期化されたFastAPIアプリ
    """
    dashboard = Dashboard(port=port, bb=blackboard, metrics_server_url=metrics_server_url)
    return dashboard.app
