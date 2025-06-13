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
                            speed: 15.0
                        });
                        
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
        # This method is called by an API endpoint and by internal polling in Vue (fetchData).
        # It should ideally fetch from a single source of truth, e.g., MetricsTracker via BlackBoard.
        metrics_to_return = {}
        try:
            if self.bb:
                # Attempt to get consolidated metrics from a single source if possible
                # For now, continue with individual param fetching as per existing logic
                # but this is where it could be simplified if MetricsTracker posts its full dict to bb.
                try:
                    entropy = await self.bb.get_param("entropy", 0.0) # Default to 0 if not set
                    target_H = await self.bb.get_param("target_H", 5.0) # Example value
                    
                    metrics_to_return = {
                        "entropy": float(entropy),
                        "vdi": float(await self.bb.get_param("vdi", 0.0)),
                        "fcr": float(await self.bb.get_param("fcr", 1.0)), # Default FCR is 1.0
                        "speed": float(await self.bb.get_param("speed", 0.0)),
                        "target_H": float(target_H) # Example, if used by controller
                    }
                    print(f"[METRICS_DEBUG] Dashboard._fetch_metrics returning from bb.get_param: {metrics_to_return}")
                    return metrics_to_return
                except Exception as e_get_param:
                    print(f"[METRICS_DEBUG] Error fetching individual params from BlackBoard: {e_get_param}. Trying pull.")
                    pass
                
                # Fallback: Try to get metrics from latest messages if direct get_param fails or is partial
                # This part of logic might be redundant if metrics are consistently stored via set_param
                # by the MetricsTracker via run_sim.py.
                messages = await self.bb.pull(k=50)
                latest_metrics = {
                    "entropy": 4.5,
                    "vdi": 0.7,
                    "fcr": 0.9,
                    "speed": 15.0
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
                        
                        # 個別のメトリクス値が含まれている場合
                        for key in ['entropy', 'vdi', 'fcr', 'speed']:
                            if key in msg and isinstance(msg[key], (int, float, str)):
                                try:
                                    latest_metrics[key] = float(msg[key])
                                except (ValueError, TypeError):
                                    pass  # 数値に変換できない場合はスキップ
                
                # This fallback logic might be simplified if metrics are reliably in bb.get_param
                # For now, keeping a simplified version of it.
                # If the above try block with get_param succeeded, this won't be reached.
                print("[METRICS_DEBUG] Dashboard._fetch_metrics: bb.get_param path failed or was incomplete, trying pull for latest_metrics.")
                messages = await self.bb.pull(k=1) # Check latest message for metrics update
                latest_metrics_from_msg = {}
                if messages and isinstance(messages[0], dict) and 'metrics' in messages[0]:
                    latest_metrics_from_msg = messages[0]['metrics']

                # Merge or select based on what was successfully fetched
                # Prioritize values from get_param if they were fetched.
                # This part is complex and depends on how metrics are stored.
                # Assuming for now that if get_param path failed, we use a default or empty.
                # The UI has its own defaults in Vue.

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
