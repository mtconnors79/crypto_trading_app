from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from datetime import datetime
from gemini_client import GeminiDashboardClient

app = FastAPI(title="Crypto Trading Bot Dashboard")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini client
gemini_client = GeminiDashboardClient()

# WebSocket connections for real-time updates
active_connections = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Send real-time updates every 10 seconds
            portfolio_data = await gemini_client.get_portfolio_summary()
            market_data = await gemini_client.get_market_data()
            
            update = {
                'type': 'portfolio_update',
                'data': portfolio_data,
                'market_data': market_data,
                'timestamp': datetime.now().isoformat()
            }
            
            await websocket.send_text(json.dumps(update))
            await asyncio.sleep(10)  # Update every 10 seconds
            
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)
        try:
            await websocket.close()
        except:
            pass

@app.get("/api/portfolio")
async def get_portfolio():
    """Get current portfolio summary"""
    return await gemini_client.get_portfolio_summary()

@app.get("/api/trades")
async def get_trades(limit: int = 50):
    """Get trade history"""
    return await gemini_client.get_trade_history(limit)

@app.get("/api/bot-analytics")
async def get_bot_analytics():
    """Get bot performance analytics"""
    return await gemini_client.get_bot_analytics()

@app.get("/api/market-data")
async def get_market_data():
    """Get current market data"""
    return await gemini_client.get_market_data()

@app.get("/api/dashboard")
async def get_dashboard_data():
    """Get all dashboard data in one call"""
    portfolio = await gemini_client.get_portfolio_summary()
    trades = await gemini_client.get_trade_history(20)
    analytics = await gemini_client.get_bot_analytics()
    market = await gemini_client.get_market_data()
    
    return {
        'portfolio': portfolio,
        'trades': trades,
        'analytics': analytics,
        'market': market,
        'timestamp': datetime.now().isoformat()
    }

@app.get("/")
async def root():
    return {"message": "Crypto Trading Bot Dashboard API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
