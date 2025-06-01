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
                const { createApp } = Vue;
                
                createApp({
                    data() {
                        return {
                            metrics: {
                                entropy: 0,
                                vdi: 0,
                                fcr: 0,
                                speed: 0
                            },
                            lambda: {
                                a: 0.3,
                                c: 0.3,
                                s: 0.1
                            },
                            blackboardMessages: [],
                            promptInput: '',
                            systemStatus: {
                                ram: 0,
                                cpu: 0
                            },
                            websocket: null,
                            chart: null
                        }
                    },
                    mounted() {
                        this.initWebSocket();
                        this.initChart();
                        this.fetchData();
                    },
                    methods: {
                        initWebSocket() {
                            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                            const url = `${protocol}//${window.location.host}/ws`;
                            
                            this.websocket = new WebSocket(url);
                            
                            this.websocket.onopen = () => {
                                console.log('WebSocket connected');
                                this.websocket.send(JSON.stringify({
                                    type: 'subscribe',
                                    channel: 'all'
                                }));
                            };
                            
                            this.websocket.onmessage = (event) => {
                                const data = JSON.parse(event.data);
                                this.handleWebSocketMessage(data);
                            };
                            
                            this.websocket.onclose = () => {
                                console.log('WebSocket disconnected');
                                setTimeout(() => this.initWebSocket(), 5000);
                            };
                        },
                        
                        handleWebSocketMessage(data) {
                            if (data.type === 'metrics') {
                                this.metrics = { ...this.metrics, ...data.data };
                                this.updateChart();
                            } else if (data.type === 'blackboard') {
                                this.blackboardMessages = data.data.messages || [];
                            }
                        },
                        
                        initChart() {
                            const ctx = document.getElementById('metricsChart').getContext('2d');
                            this.chart = new Chart(ctx, {
                                type: 'line',
                                data: {
                                    labels: [],
                                    datasets: [
                                        {
                                            label: 'Entropy',
                                            data: [],
                                            borderColor: 'rgb(75, 192, 192)',
                                            tension: 0.1
                                        },
                                        {
                                            label: 'VDI',
                                            data: [],
                                            borderColor: 'rgb(255, 99, 132)',
                                            tension: 0.1
                                        },
                                        {
                                            label: 'FCR',
                                            data: [],
                                            borderColor: 'rgb(54, 162, 235)',
                                            tension: 0.1
                                        }
                                    ]
                                },
                                options: {
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    scales: {
                                        y: {
                                            beginAtZero: true
                                        }
                                    }
                                }
                            });
                        },
                        
                        updateChart() {
                            const now = new Date().toLocaleTimeString();
                            
                            // データポイントを追加
                            this.chart.data.labels.push(now);
                            this.chart.data.datasets[0].data.push(this.metrics.entropy);
                            this.chart.data.datasets[1].data.push(this.metrics.vdi);
                            this.chart.data.datasets[2].data.push(this.metrics.fcr);
                            
                            // 最大50ポイントに制限
                            if (this.chart.data.labels.length > 50) {
                                this.chart.data.labels.shift();
                                this.chart.data.datasets.forEach(dataset => dataset.data.shift());
                            }
                            
                            this.chart.update();
                        },
                        
                        async fetchData() {
                            try {
                                // メトリクスを取得
                                const metricsResponse = await fetch('/api/metrics');
                                const metrics = await metricsResponse.json();
                                this.metrics = { ...this.metrics, ...metrics };
                                
                                // BlackBoardデータを取得
                                const blackboardResponse = await fetch('/api/blackboard');
                                const blackboard = await blackboardResponse.json();
                                this.blackboardMessages = blackboard.messages || [];
                                
                            } catch (error) {
                                console.error('Error fetching data:', error);
                            }
                        },
                        
                        async injectPrompt() {
                            if (!this.promptInput.trim()) return;
                            
                            try {
                                const response = await fetch('/api/inject', {
                                    method: 'POST',
                                    headers: {
                                        'Content-Type': 'application/json'
                                    },
                                    body: JSON.stringify({ prompt: this.promptInput })
                                });
                                
                                const result = await response.json();
                                if (result.status === 'success') {
                                    this.promptInput = '';
                                    alert('Prompt injected successfully!');
                                } else {
                                    alert('Error injecting prompt: ' + result.message);
                                }
                            } catch (error) {
                                console.error('Error injecting prompt:', error);
                                alert('Error injecting prompt');
                            }
                        }
                    }
                }).mount('#app');
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
            if self.bb:
                # BlackBoardからパラメータを取得
                lambda_a = await self.bb.get_param("λ_a") or 0.3
                lambda_c = await self.bb.get_param("λ_c") or 0.3
                lambda_s = await self.bb.get_param("λ_s") or 0.1
                
                # メッセージ数からエントロピーとVDIを推定
                messages = await self.bb.pull(k=100)
                entropy = len(messages) * 0.1 if messages else 0
                vdi = min(len(messages) / 50.0, 1.0) if messages else 0
                fcr = 0.8 + (len(messages) % 10) * 0.02  # 仮のFCR値
                speed = 10.0 + (len(messages) % 20) * 0.5  # 仮の速度値
                
                return {
                    "entropy": entropy,
                    "vdi": vdi,
                    "fcr": fcr,
                    "speed": speed,
                    "lambda_a": lambda_a,
                    "lambda_c": lambda_c,
                    "lambda_s": lambda_s
                }
            else:
                # ダミーデータ
                return {
                    "entropy": 4.5,
                    "vdi": 0.7,
                    "fcr": 0.9,
                    "speed": 15.2,
                    "lambda_a": 0.3,
                    "lambda_c": 0.3,
                    "lambda_s": 0.1
                }
        except Exception as e:
            print(f"Error fetching metrics: {e}")
            return {
                "entropy": 0,
                "vdi": 0,
                "fcr": 0,
                "speed": 0,
                "lambda_a": 0.3,
                "lambda_c": 0.3,
                "lambda_s": 0.1
            }
    
    async def _fetch_blackboard(self) -> Dict[str, Any]:
        """BlackBoardの情報を取得"""
        try:
            if self.bb:
                messages = await self.bb.pull(k=20)
                formatted_messages = []
                for i, message in enumerate(messages):
                    if isinstance(message, dict):
                        formatted_messages.append({
                            "agent_id": message.get("agent_id", "Unknown"),
                            "text": message.get("text", str(message)),
                            "timestamp": message.get("timestamp", "")
                        })
                    else:
                        formatted_messages.append({
                            "agent_id": i % 3 + 1,  # ダミーのエージェントID
                            "text": str(message),
                            "timestamp": ""
                        })
                return {"messages": formatted_messages}
            else:
                # ダミーデータ
                return {
                    "messages": [
                        {"agent_id": 1, "text": "Hello, world!", "timestamp": ""},
                        {"agent_id": 2, "text": "How are you?", "timestamp": ""}
                    ]
                }
        except Exception as e:
            print(f"Error fetching blackboard: {e}")
            return {"messages": []}
    
    async def _inject_prompt(self, prompt: str) -> Dict[str, Any]:
        """プロンプトを注入"""
        try:
            print(f"Injecting prompt: {prompt}")
            if self.bb and prompt.strip():
                # BlackBoardにプロンプトを注入（正しいシグネチャで呼び出し）
                agent_id = 999  # UIからの注入を示すための特別なID
                text = f"[INJECTED] {prompt}"
                await self.bb.push(agent_id, text)
                print(f"Prompt successfully injected to BlackBoard")
                return {"status": "success", "message": "Prompt injected successfully"}
            else:
                return {"status": "error", "message": "BlackBoard not available or empty prompt"}
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
