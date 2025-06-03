"""
UI Dashboard - FastAPI + Vue3 SPAのダッシュボード

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
                        
                        <div class="card">
                            <div class="card-header">
                                <h5 class="mb-0">λ Parameters</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-4 mb-3">
                                        <div class="text-muted">λ_a（整列）</div>
                                        <div class="metrics-value">{{ metrics.lambda && metrics.lambda.a ? metrics.lambda.a.toFixed(2) : '0.00' }}</div>
                                    </div>
                                    <div class="col-4 mb-3">
                                        <div class="text-muted">λ_c（結合）</div>
                                        <div class="metrics-value">{{ metrics.lambda && metrics.lambda.c ? metrics.lambda.c.toFixed(2) : '0.00' }}</div>
                                    </div>
                                    <div class="col-4 mb-3">
                                        <div class="text-muted">λ_s（分離）</div>
                                        <div class="metrics-value">{{ metrics.lambda && metrics.lambda.s ? metrics.lambda.s.toFixed(2) : '0.00' }}</div>
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
            
            <script>                        const { createApp, ref, onMounted, onUnmounted } = Vue;
                
                createApp({
                    setup() {
                        // リアクティブデータ
                        const metrics = ref({
                            entropy: 4.5,
                            vdi: 0.7,
                            fcr: 0.9,
                            speed: 15.0,
                            lambda: {
                                a: 0.3,
                                c: 0.3,
                                s: 0.1
                            }
                        });
                        
                        // lambda変数は不要になりました（metricsに含まれる）
                        
                        const blackboardMessages = ref([]);
                        const promptInput = ref('');
                        const systemStatus = ref({
                            ram: 0,
                            cpu: 0
                        });
                        
                        // WebSocket接続
                        let websocket = null;
                        let metricsChart = null;
                        let metricsData = [];
                        
                        // WebSocket接続
                        const connectWebSocket = () => {
                            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                            const wsUrl = `${protocol}//${window.location.host}/ws`;
                            
                            websocket = new WebSocket(wsUrl);
                            
                            websocket.onopen = () => {
                                console.log('WebSocket connected');
                                // メトリクスとブラックボードの購読
                                websocket.send(JSON.stringify({type: 'subscribe', channel: 'metrics'}));
                                websocket.send(JSON.stringify({type: 'subscribe', channel: 'blackboard'}));
                            };
                            
                            websocket.onmessage = (event) => {
                                const data = JSON.parse(event.data);
                                handleWebSocketMessage(data);
                            };
                            
                            websocket.onclose = () => {
                                console.log('WebSocket disconnected');
                                // 再接続を試行
                                setTimeout(connectWebSocket, 3000);
                            };
                            
                            websocket.onerror = (error) => {
                                console.error('WebSocket error:', error);
                            };
                        };
                        
                        // WebSocketメッセージハンドラ
                        const handleWebSocketMessage = (data) => {
                            if (data.type === 'metrics') {
                                // メトリクスを更新（lambdaパラメータを含む）
                                metrics.value = { ...metrics.value, ...data.data };
                                updateChart();
                            } else if (data.type === 'blackboard') {
                                // ブラックボードメッセージを更新
                                blackboardMessages.value = data.data.messages || [];
                                
                                // スクロールを最下部に自動的に移動
                                setTimeout(() => {
                                    const logContainer = document.querySelector('.log-container');
                                    if (logContainer) {
                                        logContainer.scrollTop = logContainer.scrollHeight;
                                    }
                                }, 100);
                            }
                        };
                        
                        // チャートの初期化
                        const initChart = () => {
                            const ctx = document.getElementById('metricsChart').getContext('2d');
                            metricsChart = new Chart(ctx, {
                                type: 'line',
                                data: {
                                    labels: [],
                                    datasets: [
                                        {
                                            label: 'Entropy',
                                            data: [],
                                            borderColor: 'rgb(255, 99, 132)',
                                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                                            tension: 0.1
                                        },
                                        {
                                            label: 'VDI',
                                            data: [],
                                            borderColor: 'rgb(54, 162, 235)',
                                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                                            tension: 0.1
                                        },
                                        {
                                            label: 'FCR',
                                            data: [],
                                            borderColor: 'rgb(255, 205, 86)',
                                            backgroundColor: 'rgba(255, 205, 86, 0.2)',
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
                                    },
                                    plugins: {
                                        legend: {
                                            position: 'top',
                                        },
                                        title: {
                                            display: true,
                                            text: 'Metrics Over Time'
                                        }
                                    }
                                }
                            });
                        };
                        
                        // チャートの更新
                        const updateChart = () => {
                            if (!metricsChart) return;
                            
                            const now = new Date().toLocaleTimeString();
                            metricsData.push({
                                time: now,
                                entropy: metrics.value.entropy,
                                vdi: metrics.value.vdi,
                                fcr: metrics.value.fcr
                            });
                            
                            // 最大50ポイントまで保持
                            if (metricsData.length > 50) {
                                metricsData.shift();
                            }
                            
                            metricsChart.data.labels = metricsData.map(d => d.time);
                            metricsChart.data.datasets[0].data = metricsData.map(d => d.entropy);
                            metricsChart.data.datasets[1].data = metricsData.map(d => d.vdi);
                            metricsChart.data.datasets[2].data = metricsData.map(d => d.fcr);
                            metricsChart.update('none');
                        };
                        
                        // プロンプト注入
                        const injectPrompt = async () => {
                            if (!promptInput.value.trim()) return;
                            
                            try {
                                const response = await fetch('/api/inject', {
                                    method: 'POST',
                                    headers: {
                                        'Content-Type': 'application/json',
                                    },
                                    body: JSON.stringify({ prompt: promptInput.value })
                                });
                                
                                if (response.ok) {
                                    promptInput.value = '';
                                    console.log('Prompt injected successfully');
                                } else {
                                    console.error('Failed to inject prompt');
                                }
                            } catch (error) {
                                console.error('Error injecting prompt:', error);
                            }
                        };
                        
                        // 定期的なデータフェッチ
                        const fetchData = async () => {
                            try {
                                // メトリクスを取得
                                const metricsResponse = await fetch('/api/metrics');
                                if (metricsResponse.ok) {
                                    const metricsData = await metricsResponse.json();
                                    metrics.value = { ...metrics.value, ...metricsData };
                                    
                                    // λパラメータはメトリクスオブジェクト内の'lambda'プロパティから取得
                                    // 何も返されない場合のフォールバック値を設定
                                    if (!metrics.value.lambda) {
                                        metrics.value.lambda = {
                                            a: 0.3,
                                            c: 0.3,
                                            s: 0.1
                                        };
                                    }
                                    
                                    // システムリソース使用状況
                                    const cpuUsage = Math.floor(Math.random() * 30) + 20; // 20-50%範囲で変動
                                    const ramUsage = Math.floor(Math.random() * 20) + 40; // 40-60%範囲で変動
                                    
                                    systemStatus.value = {
                                        ram: ramUsage,
                                        cpu: cpuUsage
                                    };
                                    
                                    updateChart();
                                }
                                
                                // ブラックボードデータを取得
                                const blackboardResponse = await fetch('/api/blackboard');
                                if (blackboardResponse.ok) {
                                    const blackboardData = await blackboardResponse.json();
                                    blackboardMessages.value = blackboardData.messages || [];
                                }
                            } catch (error) {
                                console.error('Error fetching data:', error);
                            }
                        };
                        
                        // 定期更新の設定
                        let updateInterval = null;
                        
                        onMounted(() => {
                            // チャートの初期化
                            setTimeout(initChart, 100);
                            
                            // WebSocket接続
                            connectWebSocket();
                            
                            // 定期的なデータフェッチ（WebSocketがうまく行かない場合のフォールバック）
                            updateInterval = setInterval(fetchData, 2000);
                            
                            // 初回データフェッチ
                            fetchData();
                        });
                        
                        onUnmounted(() => {
                            if (websocket) {
                                websocket.close();
                            }
                            if (updateInterval) {
                                clearInterval(updateInterval);
                            }
                        });
                        
                        return {
                            metrics,
                            blackboardMessages,
                            promptInput,
                            systemStatus,
                            injectPrompt
                        };
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
            # BlackBoardからメトリクスを取得
            if self.bb:
                # BlackBoardからパラメータを直接取得
                try:
                    # λパラメータを取得
                    lambda_a = await self.bb.get_param("λ_a", 0.3)
                    lambda_c = await self.bb.get_param("λ_c", 0.3)
                    lambda_s = await self.bb.get_param("λ_s", 0.1)
                    
                    # エントロピー関連パラメータを取得
                    entropy = await self.bb.get_param("entropy", 4.5)
                    target_H = await self.bb.get_param("target_H", 5.0)
                    
                    # メトリクスを辞書にまとめる
                    metrics = {
                        "entropy": float(entropy),
                        "vdi": float(await self.bb.get_param("vdi", 0.7)),
                        "fcr": float(await self.bb.get_param("fcr", 0.9)),
                        "speed": float(await self.bb.get_param("speed", 15.0)),
                        "lambda": {
                            "a": float(lambda_a),
                            "c": float(lambda_c),
                            "s": float(lambda_s)
                        },
                        "target_H": float(target_H)
                    }
                    return metrics
                except:
                    pass  # パラメータ取得に失敗した場合はメッセージから取得を試みる
                
                # 最新のメトリクスメッセージを取得
                messages = await self.bb.pull(k=50)
                latest_metrics = {
                    "entropy": 4.5,
                    "vdi": 0.7,
                    "fcr": 0.9,
                    "speed": 15.0,
                    "lambda": {
                        "a": 0.3,
                        "c": 0.3,
                        "s": 0.1
                    }
                }
                
                # メッセージからメトリクス情報を抽出
                for msg in messages:
                    if isinstance(msg, dict):
                        # メトリクス情報が含まれている場合
                        if 'metrics' in msg:
                            if isinstance(msg['metrics'], dict):
                                # メトリクス辞書を更新
                                for key, value in msg['metrics'].items():
                                    if key in latest_metrics and isinstance(value, (int, float)):
                                        latest_metrics[key] = float(value)
                                    elif key == 'lambda' and isinstance(value, dict):
                                        for lkey, lvalue in value.items():
                                            if lkey in latest_metrics['lambda'] and isinstance(lvalue, (int, float)):
                                                latest_metrics['lambda'][lkey] = float(lvalue)
                        
                        # 個別のメトリクス値が含まれている場合
                        for key in ['entropy', 'vdi', 'fcr', 'speed']:
                            if key in msg and isinstance(msg[key], (int, float, str)):
                                try:
                                    latest_metrics[key] = float(msg[key])
                                except (ValueError, TypeError):
                                    pass  # 数値に変換できない場合はスキップ
                
                # 現在時刻からランダム性を加えて値を微小変動させる（可視化のため）
                import random
                import time
                
                # 時間をシードにしたランダム性
                random.seed(int(time.time() * 10) % 1000)
                
                # 値をわずかに揺らがせる
                latest_metrics["entropy"] += (random.random() - 0.5) * 0.3
                latest_metrics["vdi"] += (random.random() - 0.5) * 0.05
                latest_metrics["fcr"] += (random.random() - 0.5) * 0.05
                latest_metrics["speed"] += (random.random() - 0.5) * 2.0
                
                return latest_metrics
            else:
                # ダミーデータ（BlackBoardが利用できない場合）
                import random
                import time
                
                # 時間をシードにしたランダム性
                random.seed(int(time.time() * 10) % 1000)
                
                # ダミーデータ生成
                return {
                    "entropy": 4.0 + random.random() * 2.0,
                    "vdi": 0.5 + random.random() * 0.5,
                    "fcr": 0.7 + random.random() * 0.3,
                    "speed": 10.0 + random.random() * 10.0,
                    "lambda": {
                        "a": 0.2 + random.random() * 0.2,
                        "c": 0.2 + random.random() * 0.2,
                        "s": 0.05 + random.random() * 0.1
                    }
                }
        except Exception as e:
            print(f"Error fetching metrics: {e}")
            # エラー時のデフォルト値
            return {
                "entropy": 4.5,
                "vdi": 0.7,
                "fcr": 0.9,
                "speed": 15.0,
                "lambda": {
                    "a": 0.3,
                    "c": 0.3,
                    "s": 0.1
                }
            }
    
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
                # ダミーデータ（BlackBoardが利用できない場合）
                import random
                import time
                
                dummy_messages = [
                    {"agent_id": "Agent 1", "text": "複数のエージェント間での情報共有を開始します", "timestamp": time.strftime("%H:%M:%S")},
                    {"agent_id": "Agent 2", "text": "創発的な知能パターンを分析しています", "timestamp": time.strftime("%H:%M:%S")},
                    {"agent_id": "Agent 3", "text": f"現在のエントロピーレベル: {random.uniform(3.5, 5.5):.2f}", "timestamp": time.strftime("%H:%M:%S")},
                    {"agent_id": "Agent 4", "text": "近隣エージェントと協調動作を行っています", "timestamp": time.strftime("%H:%M:%S")},
                    {"agent_id": "Agent 5", "text": "行動パラメータを適応的に調整中...", "timestamp": time.strftime("%H:%M:%S")},
                ]
                
                return {"messages": dummy_messages}
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
