# api/main.py
import os
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backtest.metrics import load_metrics
from config.settings import TIMEFRAME
from data.features import add_features
# 本地模块
from data.fetcher import fetch_ohlcv
from model.ensemble_model import EnsembleModel
from model.model_registry import (
    load_latest_model_path,
    get_model_metadata,
    get_all_versions,
    find_best_model_by_auc
)
from model.train import train_and_evaluate

# 初始化 FastAPI 应用
app = FastAPI(
    title="Butterfly Bot API",
    description="智能量化交易系统后端 API",
    version="1.0.0"
)

# 允许跨域（方便前端调试）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局缓存模型实例（避免重复加载）
_cached_model = None
_cached_model_version = None

def get_cached_ensemble_model():
    global _cached_model, _cached_model_version
    latest_path = load_latest_model_path()
    if latest_path is None:
        raise HTTPException(status_code=500, detail="未找到训练好的模型，请先运行训练。")

    current_version = os.path.basename(latest_path).replace(".pkl", "")
    if _cached_model is None or _cached_model_version != current_version:
        _cached_model = EnsembleModel(latest_path, TIMEFRAME)
        _cached_model_version = current_version

    return _cached_model

@app.get("/")
def root():
    return {"message": "欢迎使用 Butterfly Bot API", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/predict")
def get_prediction():
    """
    获取当前最新 K 线的 AI 预测信号（上涨概率）
    """
    try:
        df = fetch_ohlcv(limit=200)
        df = add_features(df)
        model = get_cached_ensemble_model()
        prob = model.predict(df)
        return {
            "timestamp": df.index[-1].isoformat(),
            "close_price": float(df["close"].iloc[-1]),
            "up_probability": round(prob, 4),
            "signal": "BUY" if prob > 0.6 else "SELL" if prob < 0.4 else "HOLD",
            "model_version": _cached_model_version,
            "weights": model.weights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.get("/ohlcv")
def get_ohlcv(limit: int = 100):
    """
    获取最近的 K 线数据（用于图表展示）
    """
    try:
        df = fetch_ohlcv(limit=limit)
        result = df.reset_index()[["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="records")
        for r in result:
            r["timestamp"] = r["timestamp"].isoformat()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取 K 线失败: {str(e)}")

@app.get("/metrics")
def get_backtest_metrics():
    """
    获取最新回测指标（AUC、胜率等）
    """
    try:
        metrics = load_metrics()
        return metrics
    except Exception as e:
        return {"error": str(e), "auc": 0.55, "win_rate": 0.5}

@app.get("/models/versions")
def list_model_versions():
    """
    列出所有模型版本
    """
    versions = get_all_versions()
    return {"versions": versions}

@app.get("/models/latest")
def get_latest_model_info():
    """
    获取当前最优模型的详细信息
    """
    latest_path = load_latest_model_path()
    if not latest_path:
        raise HTTPException(status_code=404, detail="无可用模型")
    version = os.path.basename(latest_path).replace(".pkl", "")
    meta = get_model_metadata(version)
    return {
        "version": version,
        "metadata": meta
    }

@app.post("/retrain")
def trigger_retrain():
    """
    手动触发模型重训练
    """
    try:
        version, auc = train_and_evaluate()
        best_v = find_best_model_by_auc()
        return {
            "message": "重训练完成",
            "new_version": version,
            "auc": round(auc, 4),
            "best_version": best_v
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练失败: {str(e)}")

@app.get("/strategy/signal")
def get_strategy_signal():
    """
    模拟策略当前应执行的操作（仅基于最新数据，不执行交易）
    """
    pred = get_prediction()
    return {
        "action": pred["signal"],
        "probability": pred["up_probability"],
        "price": pred["close_price"],
        "timestamp": pred["timestamp"]
    }
