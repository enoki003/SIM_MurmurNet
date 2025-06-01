#!/usr/bin/env python3
"""
Enhanced Dashboard starter script
"""

import asyncio
import uvicorn
from slm_emergent_ai.ui.dashboard_enhanced import EnhancedDashboard

async def start_enhanced_dashboard():
    """改良版ダッシュボードを起動"""
    dashboard = EnhancedDashboard(port=8001)
    
    # バックグラウンドタスクを開始
    asyncio.create_task(dashboard.run_metrics_updater(interval=0.3))
    asyncio.create_task(dashboard.run_blackboard_updater(interval=0.2))
    
    # FastAPIサーバーを起動
    config = uvicorn.Config(dashboard.app, host="0.0.0.0", port=8001, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    print("Starting Enhanced Dashboard on port 8001...")
    asyncio.run(start_enhanced_dashboard())
