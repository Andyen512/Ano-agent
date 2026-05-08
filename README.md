# 通用视频标注框架

基于 Flask 的视频标注系统，支持分步标注、用户管理、视频裁切导出等功能。

## 特性

- 用户认证 (登录/登出/用户管理)
- 视频列表和筛选 (按数据集、标注状态)
- 分步标注 (可配置步骤)
- 标注进度统计
- 视频裁切和批量导出
- 响应式布局

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.py` 文件:

```python
# 视频文件根目录
PUBLIC_DATA_ROOT = Path("/path/to/your/video/data")

# 将 video_key 解析为视频文件路径
def resolve_video_path(video_key):
    return PUBLIC_DATA_ROOT / video_key

# 定义标注步骤
ANNOTATION_STEPS = {
    "description": "正常视频描述或风险描述",
    "solution_for_person": "解决方案-对人",
    "solution_for_hazard_source": "解决方案-对危险源",
    "solution_prevent_recurrence": "解决方案-防止危险复发"
}

# 分类法定义
TAXONOMY = {
    "level1_scene": ["dining room", "kitchen", ...],
    "level2_subject": ["child", "older adult", ...],
    "level3_risk_type": ["fall/instability", ...],
}
```

### 3. 初始化数据库

首次运行会自动创建数据库表。需要预先在数据库中插入视频信息:

```sql
-- 插入视频信息
INSERT INTO videos (video_key, dataset, file_path) VALUES
('dataset/video1.mp4', 'dataset', '/path/to/video1.mp4'),
('dataset/video2.mp4', 'dataset', '/path/to/video2.mp4');

-- 插入模型预测 (可选，用于显示模型判断)
INSERT INTO model_predictions (video_key, risk, description) VALUES
('dataset/video1.mp4', 'Yes', '有风险'),
('dataset/video2.mp4', 'No', '正常');
```

### 4. 运行

```bash
python app.py
```

访问 http://localhost:5001 ，使用 `admin/admin123` 登录。

## 数据库表结构

### videos 表
```sql
video_key TEXT PRIMARY KEY,  -- 视频唯一标识
site TEXT,                    -- 来源网站
video_id TEXT,                -- 视频ID
dataset TEXT NOT NULL,        -- 数据集名称
title TEXT,                   -- 标题
file_path TEXT NOT NULL,      -- 文件路径
duration_seconds INTEGER,      -- 时长
description TEXT,              -- 描述
category TEXT,                 -- 分类
created_at TEXT               -- 创建时间
```

### model_predictions 表
```sql
video_key TEXT PRIMARY KEY,
risk TEXT,                    -- Yes/No
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
...
```

### human_annotations 表
```sql
id INTEGER PRIMARY KEY,
video_key TEXT NOT NULL,
annotator TEXT NOT NULL,     -- 标注者
risk TEXT,
risk_subtype TEXT,
level1_scene TEXT,
level2_subject TEXT,
level3_risk_type TEXT,
description TEXT,
solution_for_person TEXT,
solution_for_hazard_source TEXT,
solution_prevent_recurrence TEXT,
notes TEXT,
trim_segments TEXT,          -- 裁切时间段
updated_at TEXT,
UNIQUE(video_key, annotator)
```

## 自定义分类法

在 `config.py` 中修改 `TAXONOMY`:

```python
TAXONOMY = {
    "category1": ["option1", "option2", "option3"],
    "category2": ["optionA", "optionB", "optionC"],
}
```

前端会自动读取并生成下拉选择框。

## 自定义标注步骤

在 `config.py` 中修改 `ANNOTATION_STEPS`:

```python
ANNOTATION_STEPS = {
    "step1_id": "步骤1显示名称",
    "step2_id": "步骤2显示名称",
    "step3_id": "步骤3显示名称",
}
```

前端会根据步骤生成可点击的步骤指示器。

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/login` | POST | 登录 |
| `/api/logout` | POST | 登出 |
| `/api/me` | GET | 当前用户信息 |
| `/api/videos` | GET | 获取视频列表 |
| `/api/videos/<key>/media` | GET | 获取视频文件 |
| `/api/videos/<key>/annotation` | POST | 保存标注 |
| `/api/videos/<key>/trim` | POST | 裁切视频 |
| `/api/datasets` | GET | 获取数据集列表 |
| `/api/taxonomy` | GET | 获取分类法 |
| `/api/stats` | GET | 获取统计信息 |
| `/api/export` | GET | 导出标注数据 |
| `/api/users` | GET/POST/DELETE | 用户管理 |

## 扩展开发

### 添加自定义模型输出

在 `index.html` 的 `loadModelOutputsForVideo` 函数中添加:

```javascript
async function loadModelOutputsForVideo(videoKey) {
  try {
    const res = await fetch(`/api/videos/${encodeURIComponent(videoKey)}/model_outputs`);
    const data = await res.json();
    state.modelOutputs = data.outputs || [];
  } catch (e) {
    state.modelOutputs = [];
  }
  renderStepForm();
}
```

在后端添加对应的 API 路由。

## 目录结构

```
app_release/
├── app.py              # Flask 主应用
├── config.py           # 配置文件
├── requirements.txt    # Python 依赖
├── static/
│   ├── index.html     # 主页面
│   ├── login.html     # 登录页
│   └── styles.css     # 样式
└── README.md          # 文档
```
