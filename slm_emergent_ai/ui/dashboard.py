"""
UI Dashboard - FastAPI + Vue3 SPAのダッシュボード実装

SLM Emergent AIのUIダッシュボード。
WebSocketでmetrics/bb/streamを購読し、リアルタイムでグラフを更新。
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional
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
            
            print(f"Dashboard initialized on port {self.port}")
        except Exception as e:
            print(f"Error initializing dashboard: {e}")
    
    def _get_dashboard_html(self) -> str:
        """ダッシュボードのHTMLを取得"""
        return """
        <!DOCTYPE html>
        <html lang="en">
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
                                        <div class="text-muted">Entropy</div>
                                        <div class="metrics-value">{{ metrics.entropy.toFixed(2) }}</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">VDI</div>
                                        <div class="metrics-value">{{ metrics.vdi.toFixed(2) }}</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">FCR</div>
                                        <div class="metrics-value">{{ metrics.fcr.toFixed(2) }}</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">Speed (tok/s)</div>
                                        <div class="metrics-value">{{ metrics.speed.toFixed(2) }}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">λ Parameters</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-4 mb-3">
                                        <div class="text-muted">λ_a</div>
                                        <div class="metrics-value">{{ lambda.a.toFixed(2) }}</div>
                                    </div>
                                    <div class="col-4 mb-3">
                                        <div class="text-muted">λ_c</div>
                                        <div class="metrics-value">{{ lambda.c.toFixed(2) }}</div>
                                    </div>
                                    <div class="col-4 mb-3">
                                        <div class="text-muted">λ_s</div>
                                        <div class="metrics-value">{{ lambda.s.toFixed(2) }}</div>
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
                                        <strong>Agent {{ message.agent_id }}:</strong> {{ message.text }}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-4">
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
                // Vue.js アプリケーションの実装
                // 実際の実装では、より詳細なコードが必要
            </script>
        </body>
        </html>
        """
    
    async def _handle_websocket_message(self, websocket: WebSocket, data: str):
        """WebSocketメッセージを処理"""
        try:
            message = json.loads(data)
            if message.get("type") == "subscribe":
                # 購読チャンネルを処理
                channel = message.get("channel")
                print(f"Client subscribed to channel: {channel}")
            elif message.get("type") == "inject":
                # プロンプト注入を処理
                prompt = message.get("prompt")
                await self._inject_prompt(prompt)
        except Exception as e:
            print(f"Error handling WebSocket message: {e}")
    
    async def _fetch_metrics(self) -> Dict[str, Any]:
        """メトリクスを取得"""
        try:
            # 実際の実装では、メトリクスサーバーからデータを取得
            # ここではダミーデータを返す
            return {
                "entropy": 4.5,
                "vdi": 0.7,
                "fcr": 0.9,
                "speed": 15.2
            }
        except Exception as e:
            print(f"Error fetching metrics: {e}")
            return {}
    
    async def _fetch_blackboard(self) -> Dict[str, Any]:
        """BlackBoardの情報を取得"""
        try:
            if self.bb:
                messages = self.bb.pull(k=20)
                return {"messages": messages}
            else:
                # ダミーデータ
                return {
                    "messages": [
                        {"agent_id": 1, "text": "Hello, world!"},
                        {"agent_id": 2, "text": "How are you?"}
                    ]
                }
        except Exception as e:
            print(f"Error fetching blackboard: {e}")
            return {"messages": []}
    
    async def _inject_prompt(self, prompt: str) -> Dict[str, Any]:
        """プロンプトを注入"""
        try:
            print(f"Injecting prompt: {prompt}")
            # 実際の実装では、BlackBoardにプロンプトを注入
            return {"status": "success"}
        except Exception as e:
            print(f"Error injecting prompt: {e}")
            return {"status": "error", "message": str(e)}
    
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
