"""
示例配置 - 演示如何自定义标注框架

这是一个完整的配置示例，展示了不同的自定义方式。
复制此文件为 config.py 并根据需要修改。
"""
import os
from pathlib import Path

# ================== 基础配置 ==================

SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-change-me")

DATABASE_PATH = Path(__file__).parent / "data" / "app.db"

PORT = int(os.environ.get("PORT", "5001"))

# ================== 数据配置 ==================

# 示例 1: 简单映射 (视频文件在公共目录下)
# video_key = "dataset/video.mp4" -> "/data/videos/dataset/video.mp4"
PUBLIC_DATA_ROOT = Path("/data/videos")

def resolve_video_path(video_key):
    return PUBLIC_DATA_ROOT / video_key

# 示例 2: 复杂映射 (需要处理特殊路径)
# video_key = "s3://bucket/path/video.mp4" -> 本地缓存路径
# CACHE_DIR = Path("/tmp/video_cache")
# 
# def resolve_video_path(video_key):
#     if video_key.startswith("s3://"):
#         # 下载到本地缓存
#         local_path = CACHE_DIR / video_key.replace("s3://", "")
#         if not local_path.exists():
#             download_from_s3(video_key, local_path)
#         return local_path
#     return PUBLIC_DATA_ROOT / video_key

# ================== 分类法配置 ==================

# 示例: 简单的二分类
# TAXONOMY = {
#     "category": ["Category A", "Category B", "Category C"],
# }

# 示例: 场景分类
TAXONOMY = {
    "level1_scene": [
        "indoor", "outdoor", "vehicle", "public_space", "private_space"
    ],
    "level2_subject": [
        "adult", "child", "elderly", "pet", "vehicle_occupant"
    ],
    "level3_risk_type": [
        "collision", "fall", "fire", "electrical", "physical_hazard", "none"
    ],
}

# ================== 标注步骤配置 ==================

# 示例: 简单的两步标注
# ANNOTATION_STEPS = {
#     "category": "选择分类",
#     "notes": "备注信息"
# }

# 默认的四步标注
ANNOTATION_STEPS = {
    "description": "正常视频描述或风险描述",
    "solution_for_person": "解决方案-对人",
    "solution_for_hazard_source": "解决方案-对危险源",
    "solution_prevent_recurrence": "解决方案-防止危险复发"
}

# ================== 中文标签映射 ==================

LABEL_MAP = {
    "level1_scene": {
        "indoor": "室内", "outdoor": "室外", "vehicle": "车辆",
        "public_space": "公共场所", "private_space": "私人空间"
    },
    "level2_subject": {
        "adult": "成年人", "child": "儿童", "elderly": "老年人",
        "pet": "宠物", "vehicle_occupant": "车辆乘客"
    },
    "level3_risk_type": {
        "collision": "碰撞", "fall": "跌倒", "fire": "火灾",
        "electrical": "电气", "physical_hazard": "物理危险", "none": "无"
    },
}

def to_chinese(key, val):
    return LABEL_MAP.get(key, {}).get(val, val or "-")

# ================== 前端配置 ==================

SITE_TITLE = "视频标注平台"  # 浏览器标签页标题

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"

# ================== 数据库表结构 (高级) ==================

# 如需添加自定义字段，可以在这里修改表结构
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
