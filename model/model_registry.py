# model/model_registry.py
"""
模型注册中心：保存、加载、比较模型版本
"""

import glob
import json
import os
from datetime import datetime
from typing import Any, Optional, Tuple, List

import joblib

from config.settings import (
    REGISTRY_DIR
)

os.makedirs(REGISTRY_DIR, exist_ok=True)
LATEST_FILE = os.path.join(REGISTRY_DIR, "latest_model.txt")


def save_model_with_metadata(model, metadata: dict) -> str:
    """
    保存模型和元数据，返回版本号（如 v20251103_1620）
    """
    version = "v" + datetime.now().strftime("%Y%m%d_%H%M")
    model_path = os.path.join(REGISTRY_DIR, f"{version}.pkl")
    meta_path = os.path.join(REGISTRY_DIR, f"{version}.json")

    joblib.dump(model, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return version


def load_model_by_version(version: str) -> Tuple[Any, dict]:
    """根据版本加载模型和元数据"""
    model_path = os.path.join(REGISTRY_DIR, f"{version}.pkl")
    meta_path = os.path.join(REGISTRY_DIR, f"{version}.json")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"模型版本 {version} 不存在")

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, metadata


def list_all_versions() -> List[str]:
    """列出所有模型版本（按时间排序）"""
    pattern = os.path.join(REGISTRY_DIR, "v*.json")
    files = glob.glob(pattern)
    versions = [os.path.basename(f).replace(".json", "") for f in files]
    return sorted(versions)


def get_all_versions() -> List[str]:
    """兼容名：返回所有版本（同 list_all_versions）。"""
    return list_all_versions()


def find_best_model_by_auc() -> str:
    """找出 AUC 最高的模型版本"""
    versions = list_all_versions()
    if not versions:
        raise RuntimeError("无可用模型")

    best_version = versions[0]
    best_auc = -1.0

    for v in versions:
        _, meta = load_model_by_version(v)
        auc = meta.get("auc", 0.5)
        if auc > best_auc:
            best_auc = auc
            best_version = v

    return best_version


def update_latest_model(version: str):
    """更新 latest_model.txt 指向最优版本"""
    with open(LATEST_FILE, "w") as f:
        f.write(version)


def load_latest_model_path() -> Optional[str]:
    """返回最新（最优）模型的版本号，若无则返回 None"""
    if not os.path.exists(LATEST_FILE):
        return None
    with open(LATEST_FILE, "r") as f:
        version = f.read().strip()
    if not version:
        return None
    # 构造模型文件完整路径并验证是否存在，返回路径以便调用方直接使用
    model_path = os.path.join(REGISTRY_DIR, f"{version}.pkl")
    return model_path if os.path.exists(model_path) else None


def get_model_metadata(version: str) -> dict:
    """读取并返回指定版本的元数据（json）。

    参数 version 可以是版本名（vYYYYMMDD_HHMM）或带 .json/.pkl 的文件名。
    """
    # 规范化版本名
    v = os.path.basename(version)
    if v.endswith('.pkl'):
        v = v.replace('.pkl', '')
    if v.endswith('.json'):
        v = v.replace('.json', '')

    meta_path = os.path.join(REGISTRY_DIR, f"{v}.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"模型元数据 {v} 不存在")
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)
