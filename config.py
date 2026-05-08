"""
配置模块 - 通用视频标注框架

使用方法:
1. 复制此文件为 config.py
2. 修改配置项
3. 运行 app.py
"""
import os
from pathlib import Path

# ================== 基础配置 ==================

# Flask 密钥 (建议使用随机字符串)
SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-to-random-string")

# 数据库路径
DATABASE_PATH = Path(__file__).parent / "data" / "app.db"

# 端口
PORT = int(os.environ.get("PORT", "5001"))

# ================== 数据配置 ==================

# 视频文件根目录
PUBLIC_DATA_ROOT = Path("/data_4/liuyuan/lifebench/data/public_data")

# 视频文件路径映射函数
# 参数: video_key - 数据库中的 video_key
# 返回: 视频文件的绝对路径
# 示例: video_key = "SmartHome-Bench/videos/smartbench_0067.mp4"
#       返回: "/data_4/liuyuan/lifebench/data/public_data/SmartHome-Bench/videos/smartbench_0067.mp4"
def resolve_video_path(video_key):
    """将 video_key 解析为视频文件路径"""
    return PUBLIC_DATA_ROOT / video_key

# ================== 数据库配置 ==================

# 数据库表结构配置
DATABASE_TABLES = {
    "videos": """
        video_key TEXT PRIMARY KEY,
        site TEXT,
        video_id TEXT,
        dataset TEXT NOT NULL,
        title TEXT,
        file_path TEXT NOT NULL,
        duration_seconds INTEGER,
        description TEXT,
        category TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    """,
    "model_predictions": """
        video_key TEXT PRIMARY KEY,
        source_output_root TEXT,
        summary_json TEXT,
        risk TEXT,
        level1_scene TEXT,
        level2_subject TEXT,
        level3_risk_type TEXT,
        description TEXT,
        risk_localization TEXT,
        solution_for_person TEXT,
        solution_for_hazard_source TEXT,
        solution_prevent_recurrence TEXT,
        yes_count INTEGER,
        no_count INTEGER,
        votes_cast INTEGER,
        complete INTEGER,
        majority_json TEXT,
        agents_json TEXT,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(video_key) REFERENCES videos(video_key)
    """,
    "human_annotations": """
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_key TEXT NOT NULL,
        annotator TEXT NOT NULL,
        risk TEXT,
        risk_subtype TEXT,
        level1_scene TEXT,
        level2_subject TEXT,
        level3_risk_type TEXT,
        description TEXT,
        risk_localization TEXT,
        solution_for_person TEXT,
        solution_for_hazard_source TEXT,
        solution_prevent_recurrence TEXT,
        notes TEXT,
        trim_segments TEXT,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(video_key, annotator),
        FOREIGN KEY(video_key) REFERENCES videos(video_key)
    """
}

# ================== 分类法配置 ==================

TAXONOMY = {
    "level1_scene": ["dining room", "kitchen", "study", "balcony", "living room", "bathroom", "bedroom", "yard"],
    "level2_subject": ["child", "older adult", "young adult", "middle-aged adult", "all"],
    "level3_risk_type": [
        "fall/instability",
        "heat/fire source",
        "collision/crush injury",
        "sharp-object danger",
        "electrical safety",
        "poisoning/accidental ingestion",
        "interpersonal conflict",
        "animal attack/biosecurity risk",
        "None",
    ],
}

# ================== 标注步骤配置 ==================

# 步骤定义: id -> 标签
ANNOTATION_STEPS = {
    "description": "正常视频描述或风险描述",
    "solution_for_person": "解决方案-对人",
    "solution_for_hazard_source": "解决方案-对危险源",
    "solution_prevent_recurrence": "解决方案-防止危险复发"
}

# ================== 默认管理员 ==================

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"

# ================== 前端配置 ==================

# 站点标题
SITE_TITLE = "视频标注平台"

# 标签映射 (用于中文显示)
LABEL_MAP = {
    "risk": { "Yes": "有风险", "No": "无风险" },
    "risk_subtype": { "abnormal": "异常行为", "risk_only": "仅有异常前兆" },
    "level1_scene": {
        "dining room": "餐厅", "kitchen": "厨房", "study": "书房",
        "balcony": "阳台", "living room": "客厅", "bathroom": "浴室",
        "bedroom": "卧室", "yard": "院子"
    },
    "level2_subject": {
        "child": "儿童", "older adult": "老年人", "young adult": "年轻人",
        "middle-aged adult": "中年人", "all": "所有人"
    },
    "level3_risk_type": {
        "fall/instability": "跌倒/不稳", "heat/fire source": "热源/火灾",
        "collision/crush injury": "碰撞/挤压伤", "sharp-object danger": "锐器危险",
        "electrical safety": "电气安全", "poisoning/accidental ingestion": "中毒/误食",
        "interpersonal conflict": "人际冲突", "animal attack/biosecurity risk": "动物攻击/生物安全风险",
        "None": "无"
    },
}

def to_chinese(key, val):
    """将英文值转换为中文"""
    return LABEL_MAP.get(key, {}).get(val, val or "-")
