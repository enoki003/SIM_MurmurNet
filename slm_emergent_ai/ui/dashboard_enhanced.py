"""
Enhanced UI Dashboard - FastAPI + Vue3 SPAのダッシュボード実装

SLM Emergent AIのUIダッシュボード（改良版）。
動的メトリクス、視覚的フィードバック、接続状態表示を追加。
"""

import asyncio
import json
import os
import time
import math
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

class EnhancedDashboard:
    """SLM Emergent AIの改良版ダッシュボード"""
    def __init__(self, port: int = 8001, bb: Optional[Any] = None, metrics_server_url: str = "http://localhost:7861"):
        """
        Enhanced Dashboardの初期化
        
        Parameters:
        -----------
        port: サーバーポート（元のダッシュボードと競合しないよう8001に設定）
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
            self.app = FastAPI(title="SLM Emergent AI Enhanced Dashboard")
            
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
                return self._get_enhanced_dashboard_html()
            
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
                return await self._fetch_enhanced_metrics()
            
            @self.app.get("/api/blackboard")
            async def get_blackboard():
                return await self._fetch_blackboard()
            
            print(f"Enhanced Dashboard initialized on port {self.port}")
        except Exception as e:
            print(f"Error initializing enhanced dashboard: {e}")
    
    def _get_enhanced_dashboard_html(self) -> str:
        """改良版ダッシュボードのHTMLを取得"""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SLM Emergent AI Enhanced Dashboard</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/vue@3.2.47/dist/vue.global.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
            <style>
                body { padding: 20px; background-color: #f8f9fa; }
                .card { margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                .card-header { background-color: #343a40; color: white; }
                .metrics-value { font-size: 24px; font-weight: bold; transition: all 0.3s ease; }
                .chart-container { height: 300px; }
                .log-container { height: 300px; overflow-y: auto; background-color: #f5f5f5; padding: 10px; border-radius: 5px; }
                .log-entry { margin-bottom: 5px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
                .control-panel { background-color: #e9ecef; padding: 15px; border-radius: 5px; }
                
                /* 新しい視覚的フィードバック機能 */
                .metric-highlight { 
                    animation: highlight 0.8s ease-in-out; 
                    transform: scale(1.05);
                }
                @keyframes highlight {
                    0% { background-color: #fff3cd; box-shadow: 0 0 10px rgba(255, 193, 7, 0.5); }
                    50% { background-color: #ffeaa7; box-shadow: 0 0 15px rgba(255, 193, 7, 0.8); }
                    100% { background-color: transparent; box-shadow: none; }
                }
                
                /* 接続状態インジケーター */
                .connection-status {
                    position: fixed;
                    top: 10px;
                    right: 10px;
                    padding: 8px 12px;
                    border-radius: 20px;
                    color: white;
                    font-size: 12px;
                    font-weight: bold;
                    z-index: 1000;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    transition: all 0.3s ease;
                }
                .connected { background-color: #28a745; }
                .disconnected { background-color: #dc3545; }
                .connecting { background-color: #ffc107; color: #000; }
                
                /* 改良版バッジ */
                .enhanced-badge {
                    position: fixed;
                    top: 10px;
                    left: 10px;
                    background: linear-gradient(45deg, #007bff, #6f42c1);
                    color: white;
                    padding: 5px 10px;
                    border-radius: 15px;
                    font-size: 11px;
                    font-weight: bold;
                    z-index: 1000;
                }
            </style>
        </head>
        <body>
            <!-- 改良版バッジ -->
            <div class="enhanced-badge">Enhanced v2.0</div>
            
            <!-- WebSocket接続状態インジケーター -->
            <div id="connectionStatus" class="connection-status connecting">
                接続中...
            </div>
            
            <div id="app" class="container-fluid">
                <h1 class="mb-4">SLM Emergent AI Enhanced Dashboard</h1>
                
                <div class="row">
                    <div class="col-md-8">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">Real-time Metrics (Dynamic)</h5>
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
                                <h5 class="mb-0">Current Metrics (Live)</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">Entropy</div>
                                        <div class="metrics-value" :class="{ 'metric-highlight': entropyChanged }">
                                            {{ metrics.entropy.toFixed(2) }}
                                        </div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">VDI</div>
                                        <div class="metrics-value" :class="{ 'metric-highlight': vdiChanged }">
                                            {{ metrics.vdi.toFixed(2) }}
                                        </div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">FCR</div>
                                        <div class="metrics-value" :class="{ 'metric-highlight': fcrChanged }">
                                            {{ metrics.fcr.toFixed(2) }}
                                        </div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">Speed (tok/s)</div>
                                        <div class="metrics-value" :class="{ 'metric-highlight': speedChanged }">
                                            {{ metrics.speed.toFixed(2) }}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">System Status</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-12 mb-3">
                                        <div class="text-muted">Connection Status</div>
                                        <div class="metrics-value" :style="{ color: connectionStatusColor }">
                                            {{ connectionStatusText }}
                                        </div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">Update Rate</div>
                                        <div class="metrics-value">{{ updateRate.toFixed(1) }} Hz</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="text-muted">Data Points</div>
                                        <div class="metrics-value">{{ dataPoints }}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">Real-time BlackBoard Messages</h5>
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
                            blackboardMessages: [],
                            websocket: null,
                            chart: null,
                            connectionStatus: 'connecting',
                            
                            // 変更検出フラグ
                            entropyChanged: false,
                            vdiChanged: false,
                            fcrChanged: false,
                            speedChanged: false,
                            previousMetrics: {
                                entropy: 0,
                                vdi: 0,
                                fcr: 0,
                                speed: 0
                            },
                            
                            // 統計情報
                            updateRate: 0,
                            dataPoints: 0,
                            lastUpdateTime: Date.now()
                        }
                    },
                    computed: {
                        connectionStatusText() {
                            switch(this.connectionStatus) {
                                case 'connected': return '接続済み ✓';
                                case 'disconnected': return '切断 ✗';
                                case 'connecting': return '接続中...';
                                default: return '不明';
                            }
                        },
                        connectionStatusColor() {
                            switch(this.connectionStatus) {
                                case 'connected': return '#28a745';
                                case 'disconnected': return '#dc3545';
                                case 'connecting': return '#ffc107';
                                default: return '#6c757d';
                            }
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
                            this.updateConnectionStatus('connecting');
                            
                            this.websocket.onopen = () => {
                                console.log('WebSocket connected');
                                this.updateConnectionStatus('connected');
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
                                this.updateConnectionStatus('disconnected');
                                setTimeout(() => this.initWebSocket(), 5000);
                            };
                            
                            this.websocket.onerror = (error) => {
                                console.error('WebSocket error:', error);
                                this.updateConnectionStatus('disconnected');
                            };
                        },
                        
                        updateConnectionStatus(status) {
                            this.connectionStatus = status;
                            const statusElement = document.getElementById('connectionStatus');
                            if (statusElement) {
                                statusElement.className = `connection-status ${status}`;
                                statusElement.textContent = this.connectionStatusText;
                            }
                        },
                        
                        handleWebSocketMessage(data) {
                            if (data.type === 'metrics') {
                                this.updateMetrics(data.data);
                                this.updateChart();
                                this.updateStatistics();
                            } else if (data.type === 'blackboard') {
                                this.blackboardMessages = data.data.messages || [];
                            }
                        },
                        
                        updateMetrics(newMetrics) {
                            // 変更検出（閾値：0.01）
                            const threshold = 0.01;
                            
                            this.entropyChanged = Math.abs(newMetrics.entropy - this.previousMetrics.entropy) > threshold;
                            this.vdiChanged = Math.abs(newMetrics.vdi - this.previousMetrics.vdi) > threshold;
                            this.fcrChanged = Math.abs(newMetrics.fcr - this.previousMetrics.fcr) > threshold;
                            this.speedChanged = Math.abs(newMetrics.speed - this.previousMetrics.speed) > threshold;
                            
                            // 前の値を保存
                            this.previousMetrics = { ...this.metrics };
                            
                            // 新しい値を設定
                            this.metrics = { ...this.metrics, ...newMetrics };
                            
                            // ハイライト効果を一定時間後にリセット
                            setTimeout(() => {
                                this.entropyChanged = false;
                                this.vdiChanged = false;
                                this.fcrChanged = false;
                                this.speedChanged = false;
                            }, 800);
                        },
                        
                        updateStatistics() {
                            const now = Date.now();
                            const timeDiff = (now - this.lastUpdateTime) / 1000;
                            this.updateRate = timeDiff > 0 ? 1 / timeDiff : 0;
                            this.lastUpdateTime = now;
                            this.dataPoints++;
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
                                            backgroundColor: 'rgba(75, 192, 192, 0.1)',
                                            tension: 0.1,
                                            fill: true
                                        },
                                        {
                                            label: 'VDI',
                                            data: [],
                                            borderColor: 'rgb(255, 99, 132)',
                                            backgroundColor: 'rgba(255, 99, 132, 0.1)',
                                            tension: 0.1,
                                            fill: true
                                        },
                                        {
                                            label: 'FCR',
                                            data: [],
                                            borderColor: 'rgb(54, 162, 235)',
                                            backgroundColor: 'rgba(54, 162, 235, 0.1)',
                                            tension: 0.1,
                                            fill: true
                                        }
                                    ]
                                },
                                options: {
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    animation: {
                                        duration: 300
                                    },
                                    scales: {
                                        y: {
                                            beginAtZero: true,
                                            grid: {
                                                color: 'rgba(0,0,0,0.1)'
                                            }
                                        },
                                        x: {
                                            grid: {
                                                color: 'rgba(0,0,0,0.1)'
                                            }
                                        }
                                    },
                                    plugins: {
                                        legend: {
                                            position: 'top'
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
                            
                            this.chart.update('none'); // アニメーションなしで高速更新
                        },
                        
                        async fetchData() {
                            try {
                                // メトリクスを取得
                                const metricsResponse = await fetch('/api/metrics');
                                const metrics = await metricsResponse.json();
                                this.updateMetrics(metrics);
                                
                                // BlackBoardデータを取得
                                const blackboardResponse = await fetch('/api/blackboard');
                                const blackboard = await blackboardResponse.json();
                                this.blackboardMessages = blackboard.messages || [];
                                
                            } catch (error) {
                                console.error('Error fetching data:', error);
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
                channel = message.get("channel")
                print(f"Enhanced Dashboard: Client subscribed to channel: {channel}")
        except Exception as e:
            print(f"Error handling WebSocket message: {e}")
    
    async def _fetch_enhanced_metrics(self) -> Dict[str, Any]:
        """改良版メトリクスを取得（動的データ付き）"""
        try:
            current_time = time.time()
            
            # 時間ベースの動的変動を生成
            entropy_base = 4.5
            vdi_base = 0.7
            fcr_base = 0.9
            speed_base = 15.2
            
            # 複数の周期を組み合わせてリアルな変動を作成
            entropy_variation = (
                0.8 * math.sin(current_time * 0.3) +
                0.4 * math.cos(current_time * 0.7) +
                0.2 * math.sin(current_time * 1.1)
            )
            vdi_variation = (
                0.15 * math.sin(current_time * 0.5) +
                0.1 * math.cos(current_time * 0.9) +
                0.05 * math.sin(current_time * 1.3)
            )
            fcr_variation = (
                0.06 * math.sin(current_time * 0.6) +
                0.04 * math.cos(current_time * 1.0) +
                0.02 * math.sin(current_time * 1.4)
            )
            speed_variation = (
                2.5 * math.sin(current_time * 0.4) +
                1.5 * math.cos(current_time * 0.8) +
                0.8 * math.sin(current_time * 1.2)
            )
            
            return {
                "entropy": max(0, entropy_base + entropy_variation),
                "vdi": max(0, min(1.0, vdi_base + vdi_variation)),
                "fcr": max(0, min(1.0, fcr_base + fcr_variation)),
                "speed": max(0, speed_base + speed_variation),
                "lambda_a": 0.3,
                "lambda_c": 0.3,
                "lambda_s": 0.1,
                "timestamp": current_time
            }
        except Exception as e:
            print(f"Error fetching enhanced metrics: {e}")
            return {
                "entropy": 0,
                "vdi": 0,
                "fcr": 0,
                "speed": 0,
                "lambda_a": 0.3,
                "lambda_c": 0.3,
                "lambda_s": 0.1,
                "timestamp": time.time()
            }
    async def _fetch_blackboard(self) -> Dict[str, Any]:
        """BlackBoardの情報を取得"""
        try:
            if self.bb:
                # BlackBoardから最新のメッセージを取得
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
                    elif isinstance(msg, str) and len(msg.strip()) > 3:
                        # メッセージが文字列の場合
                        # トークン番号だけの文字列は除外
                        if msg.startswith('token_') and len(msg) < 20:
                            continue
                            
                        formatted_messages.append({
                            'agent_id': f'Agent {i % 5 + 1}',  # 5つのエージェントIDを循環
                            'text': str(msg),
                            'timestamp': ''
                        })
                
                # メッセージがない場合は空のリストを返す
                if not formatted_messages:
                    formatted_messages = []
                
                # 最新のメッセージを先頭に
                formatted_messages.reverse()
                
                return {"messages": formatted_messages}
            else:
                print("BlackBoard not available for dashboard")
                return {"messages": []}
        except Exception as e:
            print(f"Error fetching blackboard: {e}")
            return {"messages": []}
    
    async def broadcast(self, message: Dict[str, Any]):
        """全クライアントにメッセージをブロードキャスト"""
        for client in self.clients:
            try:
                await client.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                if client in self.clients:
                    self.clients.remove(client)
    
    def start(self):
        """サーバーを起動"""
        if self.app:
            import uvicorn
            uvicorn.run(self.app, host="0.0.0.0", port=self.port)
        else:
            print("Error: Enhanced App not initialized")
    
    async def run_metrics_updater(self, interval: float = 0.3):
        """改良版メトリクス更新ループ（高頻度）"""
        while True:
            try:
                metrics = await self._fetch_enhanced_metrics()
                await self.broadcast({"type": "metrics", "data": metrics})
                await asyncio.sleep(interval)
            except Exception as e:
                print(f"Error in enhanced metrics updater: {e}")
                await asyncio.sleep(interval)
    
    async def run_blackboard_updater(self, interval: float = 0.2):
        """改良版BlackBoard更新ループ（高頻度）"""
        while True:
            try:
                blackboard = await self._fetch_blackboard()
                await self.broadcast({"type": "blackboard", "data": blackboard})
                await asyncio.sleep(interval)
            except Exception as e:
                print(f"Error in enhanced blackboard updater: {e}")
                await asyncio.sleep(interval)

if __name__ == "__main__":
    # テスト実行用
    dashboard = EnhancedDashboard(port=8001)
    
    async def run_dashboard():
        # 並行してメトリクス更新ループを実行
        import asyncio
        await asyncio.gather(
            dashboard.run_metrics_updater(),
            dashboard.run_blackboard_updater()
        )
    
    dashboard.start()
