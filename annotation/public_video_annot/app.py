import json
import os
import sqlite3
import subprocess
import tempfile
import hashlib
import secrets
from pathlib import Path
from functools import wraps
from urllib.parse import quote
import time

import requests
from flask import Flask, jsonify, request, send_file, send_from_directory, session, redirect, url_for, Response
import imageio_ffmpeg

clients = []
online_users = set()
current_viewing = {}
last_update = {"video_key": None, "timestamp": 0}
video_path_cache = {}


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
PUBLIC_DATA_ROOT = Path("/home/caiqingyuan/code/lifebench/data/public_data")
RISK_SEGMENT_ROOT = PUBLIC_DATA_ROOT / "risk_segment"
GEN_DATA_ROOT = Path("/home/caiqingyuan/code/lifebench/data/generated_videos")
LEGACY_DATA_ROOT = Path("/data_4/liuyuan/lifebench/data")
CURRENT_DATA_ROOT = PROJECT_ROOT / "data"
MODEL_OUTPUTS_ROOTS = [
    PROJECT_ROOT / "public_data_filter" / "batch_outputs" / "persistent_full" / "reviews",
    PROJECT_ROOT / "public_data_filter" / "batch_outputs" / "web_20260429T170230Z" / "reviews",
    PROJECT_ROOT / "public_data_filter" / "batch_outputs" / "web_h264_segments_20260521T231616Z" / "reviews",
    PROJECT_ROOT / "public_data_filter" / "batch_outputs" / "youtube2_segments_20260524" / "reviews",
    PROJECT_ROOT / "public_data_filter" / "batch_outputs" / "gendata_20260529" / "reviews",
]

VOLCENGINE_TRANSLATE_CONFIG = {
    "access_key_id": os.environ.get("VOLCENGINE_ACCESS_KEY", "AKLTZjBlZjE3Nzc3MmU5NGRhYzlkOGUwNGE3OTViODMxZGM"),
    "secret_access_key": os.environ.get("VOLCENGINE_SECRET_KEY", "TW1ZeE1ETXpOelUxTUROak5ETTROV0ZpWVdJNU9UVXdZVEkyWmpKaU4yRQ=="),
}

trans_cache = {}


def count_risk_segment_samples():
    """Count precursor clips that should also be represented as abnormal sources."""
    if not RISK_SEGMENT_ROOT.exists():
        return 0
    return sum(1 for _ in RISK_SEGMENT_ROOT.rglob("*.mp4"))


def volcengine_translate(texts, source_lang="auto", target_lang="zh"):
    config = VOLCENGINE_TRANSLATE_CONFIG
    if not config["access_key_id"] or not config["secret_access_key"]:
        return None
    
    cache_key = f"{source_lang}:{target_lang}:{','.join(texts)}"
    if cache_key in trans_cache:
        return trans_cache[cache_key]
    
    try:
        import os as _os
        from volcenginesdktranslate20250301 import TRANSLATE20250301Api
        from volcenginesdkcore import Configuration, ApiClient
        
        _os.environ["VOLCENGINE_ACCESS_KEY"] = config["access_key_id"]
        _os.environ["VOLCENGINE_SECRET_KEY"] = config["secret_access_key"]
        
        cfg = Configuration()
        cfg.host = "open.volcengineapi.com"
        api = TRANSLATE20250301Api(ApiClient(cfg))
        
        result = api.translate_text({
            "SourceLanguage": source_lang,
            "TargetLanguage": target_lang,
            "TextList": texts
        })
        
        if "TranslationList" in result:
            translations = [t["Translation"] for t in result["TranslationList"]]
            trans_cache[cache_key] = translations
            return translations
        return None
    except Exception as e:
        print(f"Translation error: {e}")
        return None

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
        "stranger theft",
        "None",
    ],
}

app = Flask(__name__)
app.config["DATABASE"] = str(APP_DIR / "data" / "app.db")
app.secret_key = secrets.token_hex(32)


def resolve_data_path(file_path):
    path = Path(file_path).resolve()
    if path.exists():
        return path

    try:
        relative_path = path.relative_to(LEGACY_DATA_ROOT)
    except ValueError:
        return path

    mapped_path = (CURRENT_DATA_ROOT / relative_path).resolve()
    return mapped_path if mapped_path.exists() else path


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated

(APP_DIR / "data").mkdir(parents=True, exist_ok=True)


def get_db():
    db_path = APP_DIR / "data" / "app.db"
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def ensure_column(conn, table: str, column: str, definition: str) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS videos (
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
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS model_predictions (
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
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS human_annotations (
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
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(video_key, annotator),
            FOREIGN KEY(video_key) REFERENCES videos(video_key)
        )
        """
    )
    ensure_column(conn, "human_annotations", "risk_subtype", "TEXT")
    ensure_column(conn, "human_annotations", "trim_segments", "TEXT")
    ensure_column(conn, "human_annotations", "trimmed", "INTEGER DEFAULT 0")
    ensure_column(conn, "human_annotations", "precursor_start_time", "TEXT")
    ensure_column(conn, "human_annotations", "precursor_reviewed_at", "TEXT")
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS model_prediction_zh (
            video_key TEXT PRIMARY KEY,
            description TEXT,
            risk_localization TEXT,
            solution_for_person TEXT,
            solution_for_hazard_source TEXT,
            solution_prevent_recurrence TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(video_key) REFERENCES videos(video_key)
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS model_prediction_zh_agents (
            video_key TEXT,
            agent_idx INTEGER,
            model_backend TEXT,
            description_zh TEXT,
            solution_for_person_zh TEXT,
            solution_for_hazard_source_zh TEXT,
            solution_to_prevent_recurrence_zh TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (video_key, agent_idx)
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS skipped_videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_key TEXT NOT NULL,
            annotator TEXT NOT NULL,
            skipped_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(video_key, annotator),
            FOREIGN KEY(video_key) REFERENCES videos(video_key)
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # Create default admin if no users exist
    if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        c.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            ("admin", hash_password("admin123"))
        )
    conn.commit()
    conn.close()


@app.route("/login", methods=["GET"])
def login_page():
    return send_from_directory(APP_DIR / "static", "login.html")


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ? AND password_hash = ?",
        (username, hash_password(password))
    ).fetchone()
    conn.close()
    if not user:
        return jsonify({"error": "用户名或密码错误"}), 401
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    return jsonify({"success": True, "username": user["username"]})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True})


@app.route("/register", methods=["GET"])
def register_page():
    return send_from_directory("static", "register.html")


@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400
    if len(password) < 6:
        return jsonify({"error": "密码至少6位"}), 400
    if not username.isalnum():
        return jsonify({"error": "用户名只能包含字母和数字"}), 400
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hash_password(password))
        )
        conn.commit()
        user = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()
        session["user_id"] = user["id"]
        session["username"] = username
        return jsonify({"success": True})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "用户名已存在"}), 400


@app.route("/api/me")
def api_me():
    if "user_id" not in session:
        return jsonify({"logged_in": False})
    return jsonify({"logged_in": True, "username": session["username"]})


@app.route("/api/users", methods=["GET"])
@login_required
def list_users():
    conn = get_db()
    users = conn.execute("SELECT id, username, created_at FROM users ORDER BY id").fetchall()
    conn.close()
    return jsonify([dict(u) for u in users])


@app.route("/api/users", methods=["POST"])
@login_required
def add_user():
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400
    if len(password) < 6:
        return jsonify({"error": "密码至少6位"}), 400
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hash_password(password))
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "用户名已存在"}), 400
    conn.close()
    return jsonify({"success": True})


@app.route("/api/users/<int:user_id>", methods=["DELETE"])
@login_required
def delete_user(user_id):
    conn = get_db()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route("/")
@login_required
def index():
    resp = send_from_directory(APP_DIR / "static", "index.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


@app.after_request
def no_cache(response):
    if request.path.startswith("/static/") and request.path.endswith(".js"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.route("/statistics")
@login_required
def statistics_page():
    return send_from_directory(APP_DIR / "static", "statistics.html")


@app.route("/api/my_last_annotation")
def get_my_last_annotation():
    annotator = request.args.get("annotator", "").strip()
    if not annotator:
        return jsonify({"error": "annotator required"}), 400
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM human_annotations WHERE annotator = ? ORDER BY updated_at DESC LIMIT 1",
        (annotator,)
    ).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "no annotation found"}), 404
    return jsonify(dict(row))


@app.route("/api/stream")
def stream():
    username = session.get("username") or request.args.get("annotator", "")
    if username:
        online_users.add(username)
    def generate():
        global last_update, online_users
        last_seen = 0
        try:
            while True:
                if last_update["timestamp"] > last_seen:
                    last_seen = last_update["timestamp"]
                    yield f"data: refresh\n\n"
                time.sleep(1)
        except GeneratorExit:
            if username:
                online_users.discard(username)
    resp = Response(generate(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/api/online_users")
def get_online_users():
    return jsonify({"users": list(online_users)})


@app.route("/api/current_viewing", methods=["GET"])
def get_current_viewing():
    return jsonify(current_viewing)


@app.route("/api/current_viewing", methods=["POST"])
@login_required
def set_current_viewing():
    global current_viewing
    username = session.get("username", "")
    data = request.json or {}
    if not data:
        try:
            data = json.loads(request.data)
        except:
            data = {}
    video_key = data.get("video_key", "")
    if video_key:
        current_viewing[username] = video_key
    elif username in current_viewing:
        del current_viewing[username]
    broadcast_refresh()
    return jsonify({"success": True})


@app.route("/api/report_broken", methods=["POST"])
@login_required
def report_broken():
    data = request.json or {}
    video_key = data.get("video_key", "")
    if not video_key:
        return jsonify({"error": "缺少video_key"}), 400
    conn = get_db()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS broken_videos (id INTEGER PRIMARY KEY AUTOINCREMENT, video_key TEXT NOT NULL UNIQUE, reported_by TEXT, reported_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO broken_videos (video_key, reported_by) VALUES (?, ?)",
            (video_key, session.get("username", ""))
        )
        conn.commit()
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500
    conn.close()
    return jsonify({"success": True})


@app.route("/api/broken_videos")
@login_required
def get_broken_videos():
    conn = get_db()
    try:
        rows = conn.execute("SELECT * FROM broken_videos ORDER BY reported_at DESC").fetchall()
    except:
        rows = []
    conn.close()
    return jsonify([dict(r) for r in rows])


def broadcast_refresh():
    global last_update
    last_update = {"timestamp": time.time()}


@app.route("/api/videos")
@login_required
def get_videos():
    dataset = request.args.get("dataset", "")
    video_type = request.args.get("video_type", "")
    risk_subtype_filter = request.args.get("risk_subtype", "")
    status = request.args.get("status", "")
    duration = request.args.get("duration", "")
    sort_by = request.args.get("sort_by", "")  # 新增：排序方式
    revisit = request.args.get("revisit", "")  # 仅在异常行为-已标注下生效: '' | 'reviewed' | 'unreviewed'
    username = session.get("username", "")
    search = request.args.get("search", "")
    limit = int(request.args.get("limit", "50"))
    offset = int(request.args.get("offset", "0"))

    conn = get_db()
    c = conn.cursor()
    query = """
        SELECT v.*, p.risk AS model_risk, p.level1_scene AS model_level1_scene,
               p.level2_subject AS model_level2_subject, p.level3_risk_type AS model_level3_risk_type,
               COALESCE(z.description, p.description) AS model_description,
               COALESCE(z.risk_localization, p.risk_localization) AS model_risk_localization,
               COALESCE(z.solution_for_person, p.solution_for_person) AS model_solution_for_person,
               COALESCE(z.solution_for_hazard_source, p.solution_for_hazard_source) AS model_solution_for_hazard_source,
               COALESCE(z.solution_prevent_recurrence, p.solution_prevent_recurrence) AS model_solution_prevent_recurrence,
               p.yes_count, p.no_count, p.votes_cast, p.complete,
                p.majority_json, p.agents_json, p.summary_json,
                 EXISTS (SELECT 1 FROM skipped_videos sv_sub WHERE sv_sub.video_key = v.video_key) AS is_skipped,
                 COUNT(*) OVER() AS total_count
         FROM videos v
        LEFT JOIN model_predictions p ON p.video_key = v.video_key
        LEFT JOIN model_prediction_zh z ON z.video_key = v.video_key
        WHERE 1=1
    """
    params = []
    if dataset:
        query += " AND v.dataset = ?"
        params.append(dataset)
    elif video_type == "generated":
        query += " AND v.dataset = 'generated_videos'"
    elif video_type == "real":
        query += " AND v.dataset != 'generated_videos'"
    if duration == "short":
        query += " AND v.duration_seconds <= 60"
    elif duration == "long":
        query += " AND v.duration_seconds > 60"
    if search:
        query += " AND v.video_key LIKE ?"
        params.append(f"%{search}%")
    
    # 生成视频按分数排序优先
    if sort_by == "score_asc" and dataset == "generated_videos":
        query += " AND NOT EXISTS (SELECT 1 FROM skipped_videos sv WHERE sv.video_key = v.video_key)"
        query += """
        ORDER BY
          CAST(json_extract(p.majority_json, '$.avg_scores.avg_total_score') AS REAL) ASC,
          v.video_key
        LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
    elif sort_by == "score_desc" and dataset == "generated_videos":
        query += " AND NOT EXISTS (SELECT 1 FROM skipped_videos sv WHERE sv.video_key = v.video_key)"
        query += """
        ORDER BY
          CAST(json_extract(p.majority_json, '$.avg_scores.avg_total_score') AS REAL) DESC,
          v.video_key
        LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
    elif status == "risk":
        query += " AND p.risk = 'Yes'"
        query += " AND NOT EXISTS (SELECT 1 FROM skipped_videos sv WHERE sv.video_key = v.video_key)"
        query += " AND v.video_key NOT IN (SELECT video_key FROM human_annotations)"
        query += " ORDER BY COALESCE(p.yes_count, 0) DESC, COALESCE(p.votes_cast, 0) DESC, COALESCE(p.no_count, 0) ASC, v.dataset, v.video_key LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    elif status == "normal":
        query += " AND p.risk = 'No'"
        query += " AND NOT EXISTS (SELECT 1 FROM skipped_videos sv WHERE sv.video_key = v.video_key)"
        query += " AND v.video_key NOT IN (SELECT video_key FROM human_annotations)"
        query += " ORDER BY COALESCE(p.yes_count, 0) DESC, COALESCE(p.votes_cast, 0) DESC, COALESCE(p.no_count, 0) ASC, v.dataset, v.video_key LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    elif status == "problem":
        query += " AND p.yes_count > 0 AND p.no_count > 0"
        query += " AND NOT EXISTS (SELECT 1 FROM skipped_videos sv WHERE sv.video_key = v.video_key)"
    elif status == "pending":
        query += " AND v.video_key NOT IN (SELECT video_key FROM human_annotations)"
        query += " AND NOT EXISTS (SELECT 1 FROM skipped_videos sv WHERE sv.video_key = v.video_key)"
        query += " ORDER BY COALESCE(p.yes_count, 0) DESC, COALESCE(p.votes_cast, 0) DESC, v.dataset, v.video_key LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    elif status == "need_trim":
        query += " AND EXISTS (SELECT 1 FROM human_annotations h WHERE h.video_key = v.video_key AND h.trim_segments IS NOT NULL AND h.trim_segments != '' AND (h.trimmed IS NULL OR h.trimmed = 0))"
        query += " AND NOT EXISTS (SELECT 1 FROM skipped_videos sv WHERE sv.video_key = v.video_key)"
        query += " ORDER BY v.dataset, v.video_key LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    elif status == "trimmed":
        query += " AND EXISTS (SELECT 1 FROM human_annotations h WHERE h.video_key = v.video_key AND h.trimmed = 1)"
        query += " AND NOT EXISTS (SELECT 1 FROM skipped_videos sv WHERE sv.video_key = v.video_key)"
        query += " ORDER BY v.dataset, v.video_key LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    elif status == "annotated":
        # Matches videos that have a direct human_annotation, OR
        # web_h264_segments whose parent web_h264 video has a trim_segments
        # annotation (the segments "inherit" the parent's annotated status).
        query += """
         AND (
           EXISTS (SELECT 1 FROM human_annotations h WHERE h.video_key = v.video_key)
           OR (
             v.video_key LIKE 'web_h264_segments/%'
             AND NOT EXISTS (SELECT 1 FROM human_annotations h_self WHERE h_self.video_key = v.video_key)
             AND EXISTS (
               SELECT 1 FROM human_annotations ha
               WHERE ha.video_key LIKE 'web_h264/' || SUBSTR(v.video_key, 19, CASE WHEN INSTR(SUBSTR(v.video_key, 19), '_seg') > 0 THEN INSTR(SUBSTR(v.video_key, 19), '_seg') - 1 WHEN INSTR(SUBSTR(v.video_key, 19), '_trimmed_') > 0 THEN INSTR(SUBSTR(v.video_key, 19), '_trimmed_') - 1 ELSE LENGTH(SUBSTR(v.video_key, 19)) END) || '%'
               AND ha.trim_segments IS NOT NULL AND ha.trim_segments != ''
             )
           )
         )
        """
        # Suppress web_h264 originals that are marked for trimming and
        # whose segments exist – the segments are shown instead.
        query += """
         AND NOT (
           v.video_key LIKE 'web_h264/%'
           AND EXISTS (
             SELECT 1 FROM human_annotations ha
             WHERE ha.video_key = v.video_key
             AND ha.trim_segments IS NOT NULL AND ha.trim_segments != ''
           )
           AND EXISTS (
              SELECT 1 FROM videos seg
              WHERE seg.video_key LIKE REPLACE(REPLACE(v.video_key, 'web_h264/', 'web_h264_segments/'), '.mp4', '') || '_%'
            )
         )
        """
        if risk_subtype_filter == "abnormal":
            # Use the LATEST annotation's risk to classify.
            query += """
             AND (
               (
                 SELECT risk || ':' || risk_subtype FROM human_annotations
                 WHERE video_key = v.video_key
                 ORDER BY updated_at DESC LIMIT 1
               ) = 'Yes:abnormal'
               OR (
                 v.video_key LIKE 'web_h264_segments/%'
                 AND NOT EXISTS (SELECT 1 FROM human_annotations WHERE video_key = v.video_key)
                 AND (
                   SELECT risk || ':' || risk_subtype FROM human_annotations
                   WHERE video_key LIKE 'web_h264/' || SUBSTR(v.video_key, 19, CASE WHEN INSTR(SUBSTR(v.video_key, 19), '_seg') > 0 THEN INSTR(SUBSTR(v.video_key, 19), '_seg') - 1 WHEN INSTR(SUBSTR(v.video_key, 19), '_trimmed_') > 0 THEN INSTR(SUBSTR(v.video_key, 19), '_trimmed_') - 1 ELSE LENGTH(SUBSTR(v.video_key, 19)) END) || '%'
                   ORDER BY updated_at DESC LIMIT 1
                 ) = 'Yes:abnormal'
               )
             )
            """
        elif risk_subtype_filter == "risk_only":
            query += """
             AND (
               (
                 SELECT risk || ':' || risk_subtype FROM human_annotations
                 WHERE video_key = v.video_key
                 ORDER BY updated_at DESC LIMIT 1
               ) = 'Yes:risk_only'
               OR (
                 v.video_key LIKE 'web_h264_segments/%'
                 AND NOT EXISTS (SELECT 1 FROM human_annotations WHERE video_key = v.video_key)
                 AND (
                   SELECT risk || ':' || risk_subtype FROM human_annotations
                   WHERE video_key LIKE 'web_h264/' || SUBSTR(v.video_key, 19, CASE WHEN INSTR(SUBSTR(v.video_key, 19), '_seg') > 0 THEN INSTR(SUBSTR(v.video_key, 19), '_seg') - 1 WHEN INSTR(SUBSTR(v.video_key, 19), '_trimmed_') > 0 THEN INSTR(SUBSTR(v.video_key, 19), '_trimmed_') - 1 ELSE LENGTH(SUBSTR(v.video_key, 19)) END) || '%'
                   ORDER BY updated_at DESC LIMIT 1
                 ) = 'Yes:risk_only'
               )
             )
            """
        elif risk_subtype_filter == "abnormal_transition":
            query += """
             AND (
               (
                 SELECT risk || ':' || risk_subtype || ':' || COALESCE(precursor_start_time, '') FROM human_annotations
                 WHERE video_key = v.video_key
                 ORDER BY updated_at DESC LIMIT 1
               ) LIKE 'Yes:abnormal:%'
               AND EXISTS (
                 SELECT 1 FROM human_annotations h2 WHERE h2.video_key = v.video_key
                 AND h2.precursor_start_time IS NOT NULL AND h2.precursor_start_time != ''
               )
               OR (
                 v.video_key LIKE 'web_h264_segments/%'
                 AND NOT EXISTS (SELECT 1 FROM human_annotations WHERE video_key = v.video_key)
                 AND (
                   SELECT risk || ':' || risk_subtype FROM human_annotations
                   WHERE video_key LIKE 'web_h264/' || SUBSTR(v.video_key, 19, CASE WHEN INSTR(SUBSTR(v.video_key, 19), '_seg') > 0 THEN INSTR(SUBSTR(v.video_key, 19), '_seg') - 1 WHEN INSTR(SUBSTR(v.video_key, 19), '_trimmed_') > 0 THEN INSTR(SUBSTR(v.video_key, 19), '_trimmed_') - 1 ELSE LENGTH(SUBSTR(v.video_key, 19)) END) || '%'
                   ORDER BY updated_at DESC LIMIT 1
                 ) = 'Yes:abnormal'
                 AND EXISTS (
                   SELECT 1 FROM human_annotations ha WHERE ha.video_key LIKE 'web_h264/' || SUBSTR(v.video_key, 19, CASE WHEN INSTR(SUBSTR(v.video_key, 19), '_seg') > 0 THEN INSTR(SUBSTR(v.video_key, 19), '_seg') - 1 WHEN INSTR(SUBSTR(v.video_key, 19), '_trimmed_') > 0 THEN INSTR(SUBSTR(v.video_key, 19), '_trimmed_') - 1 ELSE LENGTH(SUBSTR(v.video_key, 19)) END) || '%'
                   AND ha.precursor_start_time IS NOT NULL AND ha.precursor_start_time != ''
                 )
               )
             )
            """
            # If the current user has personally marked this video as risk='No',
            # hide it from their "abnormal" list regardless of other annotators.
            query += """
             AND NOT (
               EXISTS (
                 SELECT 1 FROM human_annotations h_own
                 WHERE h_own.video_key = v.video_key
                 AND h_own.annotator = ?
                 AND h_own.risk = 'No'
               )
               OR (
                 v.video_key LIKE 'web_h264_segments/%'
                 AND EXISTS (
                   SELECT 1 FROM human_annotations h_own2
                   WHERE h_own2.video_key LIKE 'web_h264/' || SUBSTR(v.video_key, 19, CASE WHEN INSTR(SUBSTR(v.video_key, 19), '_seg') > 0 THEN INSTR(SUBSTR(v.video_key, 19), '_seg') - 1 WHEN INSTR(SUBSTR(v.video_key, 19), '_trimmed_') > 0 THEN INSTR(SUBSTR(v.video_key, 19), '_trimmed_') - 1 ELSE LENGTH(SUBSTR(v.video_key, 19)) END) || '%'
                   AND h_own2.annotator = ?
                   AND h_own2.risk = 'No'
                 )
               )
             )
            """
            params.extend([username, username])
        elif risk_subtype_filter == "risk_only":
            query += """
             AND (
               EXISTS (SELECT 1 FROM human_annotations h2 WHERE h2.video_key = v.video_key AND h2.risk = 'Yes' AND h2.risk_subtype = 'risk_only')
               OR (
                 v.video_key LIKE 'web_h264_segments/%'
                 AND EXISTS (SELECT 1 FROM human_annotations ha
                   WHERE ha.video_key = 'web_h264/' || SUBSTR(v.video_key, 19, CASE WHEN INSTR(SUBSTR(v.video_key, 19), '_seg') > 0 THEN INSTR(SUBSTR(v.video_key, 19), '_seg') - 1 WHEN INSTR(SUBSTR(v.video_key, 19), '_trimmed_') > 0 THEN INSTR(SUBSTR(v.video_key, 19), '_trimmed_') - 1 ELSE LENGTH(SUBSTR(v.video_key, 19)) END) || '%'
                   AND ha.risk = 'Yes' AND ha.risk_subtype = 'risk_only')
               )
             )
            """
        elif risk_subtype_filter == "abnormal_transition":
            query += """
             AND (
               EXISTS (SELECT 1 FROM human_annotations h2 WHERE h2.video_key = v.video_key AND h2.risk = 'Yes' AND h2.risk_subtype = 'abnormal' AND h2.precursor_start_time IS NOT NULL AND h2.precursor_start_time != '')
               OR (
                 v.video_key LIKE 'web_h264_segments/%'
                 AND EXISTS (SELECT 1 FROM human_annotations ha
                   WHERE ha.video_key = 'web_h264/' || SUBSTR(v.video_key, 19, CASE WHEN INSTR(SUBSTR(v.video_key, 19), '_seg') > 0 THEN INSTR(SUBSTR(v.video_key, 19), '_seg') - 1 WHEN INSTR(SUBSTR(v.video_key, 19), '_trimmed_') > 0 THEN INSTR(SUBSTR(v.video_key, 19), '_trimmed_') - 1 ELSE LENGTH(SUBSTR(v.video_key, 19)) END) || '%'
                   AND ha.risk = 'Yes' AND ha.risk_subtype = 'abnormal' AND ha.precursor_start_time IS NOT NULL AND ha.precursor_start_time != '')
               )
             )
            """
        # For the "abnormal" revisit flow, surface videos the current user
        # has NOT re-reviewed for precursor yet (precursor_reviewed_at IS NULL
        # for this annotator) at the top, with already-reviewed ones pushed
        # to the bottom ordered by review time ascending.
        if risk_subtype_filter == "abnormal":
            # A video is "已标" (reviewed) when EITHER:
            #   (a) any annotator has filled in a non-empty precursor_start_time, OR
            #   (b) any annotator has skipped it.
            # Progress is global (shared across all users).
            # Order: unreviewed first, then reviewed by earliest event time
            # (precursor fill or skip).
            reviewed_predicate = """
              (
                EXISTS (
                  SELECT 1 FROM human_annotations h3
                  WHERE h3.video_key = v.video_key
                  AND h3.precursor_start_time IS NOT NULL AND h3.precursor_start_time != ''
                ) OR EXISTS (
                  SELECT 1 FROM skipped_videos s2
                  WHERE s2.video_key = v.video_key
                )
              )
            """
            if revisit == "reviewed":
                query += " AND " + reviewed_predicate
            elif revisit == "unreviewed":
                query += " AND NOT " + reviewed_predicate
            # In the "reviewed" tab show most-recently-reviewed first;
            # otherwise reviewed group (bottom) is oldest-first.
            review_dir = "DESC" if revisit == "reviewed" else "ASC"
            query += """
             ORDER BY
               CASE WHEN """ + reviewed_predicate + """ THEN 1 ELSE 0 END ASC,
               COALESCE(
                 (
                   SELECT MIN(h4.updated_at) FROM human_annotations h4
                   WHERE h4.video_key = v.video_key
                   AND h4.precursor_start_time IS NOT NULL AND h4.precursor_start_time != ''
                 ),
                 (
                   SELECT MIN(s3.skipped_at) FROM skipped_videos s3
                   WHERE s3.video_key = v.video_key
                 )
               ) """ + review_dir + """,
               (SELECT MAX(h.updated_at) FROM human_annotations h WHERE h.video_key = v.video_key) DESC
             LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
        else:
            query += " ORDER BY (SELECT MAX(h.updated_at) FROM human_annotations h WHERE h.video_key = v.video_key) DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
    elif status == "skipped":
        query += " AND EXISTS (SELECT 1 FROM skipped_videos sv WHERE sv.video_key = v.video_key)"
        query += " ORDER BY (SELECT MAX(skipped_at) FROM skipped_videos WHERE video_key = v.video_key) DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    else:
        if not search:
            query += " AND NOT EXISTS (SELECT 1 FROM skipped_videos sv WHERE sv.video_key = v.video_key)"
        query += """
        ORDER BY
          COALESCE(p.yes_count, 0) DESC,
          COALESCE(p.votes_cast, 0) DESC,
          COALESCE(p.no_count, 0) ASC,
          v.dataset,
          v.video_key
        LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
    rows = [dict(row) for row in c.execute(query, params).fetchall()]

    for row in rows:
        row["media_url"] = f"/api/videos/{quote(row['video_key'], safe='/')}/media"
        row["majority_vote"] = json.loads(row.pop("majority_json") or "{}")
        row["qwen_summary_meta"] = json.loads(row.pop("agents_json") or "{}")
        row["viewing_users"] = [u for u, vk in current_viewing.items() if vk == row["video_key"]]
        # Load all annotations for this video
        all_anns = c.execute(
            "SELECT * FROM human_annotations WHERE video_key = ? ORDER BY updated_at DESC",
            (row["video_key"],),
        ).fetchall()
        row["annotations"] = [dict(a) for a in all_anns]
        # Prefer the current user's own annotation for editing. If this
        # user has not annotated the video yet, seed the form with the latest
        # human annotation so collaborators see each other's saved work.
        row["annotation"] = dict(all_anns[0]) if all_anns else None
        if username:
            for a in all_anns:
                if a["annotator"] == username:
                    row["annotation"] = dict(a)
                    break
        
        # 检查是否有裁切后的视频
        row["trimmed_media_url"] = None
        for a in all_anns:
            a_dict = dict(a)
            if a_dict.get("trimmed") == 1 and a_dict.get("trim_segments"):
                # 计算裁切后的文件路径
                file_path = resolve_data_path(row["file_path"])
                try:
                    relative_path = file_path.relative_to(PUBLIC_DATA_ROOT.resolve())
                    segments = parse_trim_segments(a_dict["trim_segments"])
                    if segments:
                        stem = file_path.stem
                        suffix = file_path.suffix
                        trimmed_name = _trimmed_filename(stem, segments, suffix)
                        trimmed_path = BATCH_TRIM_OUTPUT / relative_path.parent / trimmed_name
                        if trimmed_path.exists():
                            row["trimmed_media_url"] = f"/api/trimmed_media/{quote(str(relative_path.parent / trimmed_name), safe='/')}"
                        else:
                            app.logger.warning(f"Trimmed file not found: {trimmed_path}")
                    else:
                        app.logger.warning(f"Segment parse failed: {a_dict['trim_segments']}")
                except ValueError:
                    app.logger.warning(f"relative_to failed: file={file_path} root={PUBLIC_DATA_ROOT.resolve()}")
                    pass
                break

    conn.close()
    total_count = rows[0]["total_count"] if rows else 0
    return jsonify({"videos": rows, "total": total_count})


@app.route("/api/taxonomy")
@login_required
def get_taxonomy():
    return jsonify(TAXONOMY)


def load_translated_output(video_key):
    """Load pre-translated model outputs for a video."""
    # 首先从数据库的 model_prediction_zh_agents 表加载每个agent的翻译
    conn = get_db()
    rows = conn.execute(
        "SELECT agent_idx, description_zh, solution_for_person_zh, solution_for_hazard_source_zh, solution_to_prevent_recurrence_zh FROM model_prediction_zh_agents WHERE video_key = ?",
        (video_key,)
    ).fetchall()
    conn.close()
    
    if rows:
        translated = {}
        for row in rows:
            agent_idx = str(row["agent_idx"])
            translated[agent_idx] = {
                "risk_description": row["description_zh"],
                "normal_video_description": row["description_zh"],
                "solution_for_person": row["solution_for_person_zh"],
                "solution_for_hazard_source": row["solution_for_hazard_source_zh"],
                "solution_to_prevent_recurrence": row["solution_to_prevent_recurrence_zh"],
            }
        return translated
    
    # 如果新表没有数据，尝试从文件加载（per-agent 翻译）
    translated = _load_translated_from_file(video_key)
    if translated:
        return translated
    
    # 最后回退到旧表（单条翻译，所有 agent 相同）
    conn = get_db()
    row = conn.execute(
        "SELECT description, solution_for_person, solution_for_hazard_source, solution_prevent_recurrence FROM model_prediction_zh WHERE video_key = ?",
        (video_key,)
    ).fetchone()
    conn.close()
    
    if row and row["description"]:
        translated = {}
        for i in range(5):
            translated[str(i)] = {
                "risk_description": row["description"],
                "normal_video_description": row["description"],
                "solution_for_person": row["solution_for_person"],
                "solution_for_hazard_source": row["solution_for_hazard_source"],
                "solution_to_prevent_recurrence": row["solution_prevent_recurrence"],
            }
        return translated
    return None


def _load_translated_from_file(video_key):
    """从文件系统加载 per-agent 翻译。"""
    translated_base = PROJECT_ROOT / "public_data_filter" / "batch_outputs" / "translated"
    if not translated_base.exists():
        return None
    for dataset_dir in translated_base.iterdir():
        if not dataset_dir.is_dir():
            continue
        if video_key.startswith(dataset_dir.name + '/'):
            rest_path = video_key[len(dataset_dir.name)+1:]
            parts = rest_path.split('/')
            last = parts[-1]
            for ext in ('.mp4', '.avi', '.mkv', '.mov', '.webm'):
                if last.lower().endswith(ext):
                    last = last[:-len(ext)]
                    break
            parts[-1] = last
            trans_file = dataset_dir / '/'.join(parts) / "translation.json"
            if trans_file.exists():
                try:
                    with open(trans_file) as f:
                        data = json.load(f)
                    return data.get("translations", {})
                except Exception:
                    pass
    return None


def load_model_outputs(video_key):
    """Load model outputs (English and Chinese) for a video."""
    # 处理 web_h264_segments 映射到 web_h264 目录
    actual_video_key = video_key
    if video_key.startswith("web_h264_segments/"):
        actual_video_key = "web_h264/" + video_key[len("web_h264_segments/"):]
    
    # 从数据库获取原始 prompt（用于生成视频）
    original_prompt = ""
    original_prompt_zh = ""
    if video_key.startswith("generated_videos/"):
        conn = get_db()
        row = conn.execute(
            "SELECT agents_json FROM model_predictions WHERE video_key = ?",
            (video_key,)
        ).fetchone()
        conn.close()
        if row and row["agents_json"]:
            try:
                agents_data = json.loads(row["agents_json"])
                original_prompt = agents_data.get("original_prompt", "")
                original_prompt_zh = agents_data.get("original_prompt_zh", "")
            except:
                pass
    
    for reviews_dir in MODEL_OUTPUTS_ROOTS:
        if not reviews_dir.exists():
            continue
        for dataset_dir in reviews_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            if not actual_video_key.startswith(dataset_dir.name):
                continue
            rest_path = actual_video_key[len(dataset_dir.name)+1:]
            parts = rest_path.split('/')
            last = parts[-1]
            for ext in ('.mp4', '.avi', '.mkv', '.mov', '.webm'):
                if last.lower().endswith(ext):
                    last = last[:-len(ext)]
                    break
            parts[-1] = last
            review_dir = dataset_dir.joinpath(*parts)
            review_file = review_dir / "review_summary.repaired.json"
            if not review_file.exists():
                review_file = review_dir / "review_summary.json"
            if not review_file.exists():
                continue
            try:
                with open(review_file) as f:
                    data = json.load(f)
                translated = load_translated_output(video_key)
                majority_vote = data.get("majority_vote", {})
                avg_scores = majority_vote.get("avg_scores") if majority_vote else None
                outputs = []
                for agent_idx, agent in enumerate(data.get("agent_results", [])):
                    pd = agent.get("parsed_decision", {})
                    response = agent.get("response", "")
                    
                    # 检查是否是 gendata_v2 格式
                    is_gendata_v2 = "scene_match:" in response and "subject_match:" in response
                    
                    if is_gendata_v2:
                        # 解析 gendata_v2 格式的响应
                        explanation = ""
                        for line in response.split("\n"):
                            if line.strip().lower().startswith("explanation:"):
                                explanation = line.split(":", 1)[1].strip()
                                break
                        desc = explanation
                        sol_person = ""
                        sol_hazard = ""
                        sol_prev = ""
                    else:
                        # 旧格式
                        desc = pd.get("risk_description") or pd.get("normal_video_description", "")
                        sol_person = pd.get("solution_for_person", "")
                        sol_hazard = pd.get("solution_for_hazard_source", "")
                        sol_prev = pd.get("solution_to_prevent_recurrence", "")

                    trans_agent = translated.get(str(agent_idx), {}) if translated else {}

                    outputs.append({
                        "model": agent.get("model", {}).get("backend", "unknown"),
                        "description": desc,
                        "description_zh": trans_agent.get("normal_video_description") or trans_agent.get("risk_description", ""),
                        "solution_for_person": sol_person,
                        "solution_for_person_zh": trans_agent.get("solution_for_person", ""),
                        "solution_for_hazard_source": sol_hazard,
                        "solution_for_hazard_source_zh": trans_agent.get("solution_for_hazard_source", ""),
                        "solution_to_prevent_recurrence": sol_prev,
                        "solution_to_prevent_recurrence_zh": trans_agent.get("solution_to_prevent_recurrence", ""),
                        "avg_scores": avg_scores if is_gendata_v2 else None,
                        "original_prompt": original_prompt,
                        "original_prompt_zh": original_prompt_zh,
                    })
                return outputs
            except Exception as e:
                print(f"Error loading model outputs for {video_key}: {e}")
                pass
    
    # 如果没有找到 review_summary.json，但有原始 prompt，返回一个包含原始 prompt 的空 outputs
    if original_prompt:
        return [{"model": "", "original_prompt": original_prompt, "original_prompt_zh": original_prompt_zh}]
    
    return None


def load_chinese_outputs(video_key):
    """Load pre-translated Chinese outputs for a video from the Chinese schema files."""
    chinese_file = PROJECT_ROOT / "public_data_filter" / "batch_outputs" / "persistent_full" / "strict_schema_chinese_text_remaining.json"
    web_h264_chinese = PROJECT_ROOT / "public_data_filter" / "batch_outputs" / "persistent_full" / "web_h264_chinese.json"
    
    for ch_file in [chinese_file, web_h264_chinese]:
        if not ch_file.exists():
            continue
        try:
            with open(ch_file) as f:
                data = json.load(f)
            for r in data.get("results", []):
                if r.get("video_relpath") == video_key:
                    return {
                        "description_zh": r.get("Normal-video description or risk description zh", ""),
                        "solution_for_person_zh": r.get("Solutions-For person zh", ""),
                        "solution_for_hazard_source_zh": r.get("Solutions-For hazard source zh", ""),
                        "solution_prevent_recurrence_zh": r.get("Solutions-Prevent recurrence zh", ""),
                    }
        except Exception:
            continue
    return None


@app.route("/api/videos/<path:video_key>/skip", methods=["POST"])
@login_required
def skip_video(video_key):
    username = session.get("username", "")
    conn = get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO skipped_videos (video_key, annotator) VALUES (?, ?)",
            (video_key, username)
        )
        # Mark the user's "precursor review" as touched so this video gets
        # pushed to the back of the revisit list.
        from datetime import datetime, timezone, timedelta
        bj_now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "UPDATE human_annotations SET precursor_reviewed_at = ? "
            "WHERE video_key = ? AND annotator = ?",
            (bj_now, video_key, username),
        )
        conn.commit()
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500
    conn.close()
    return jsonify({"success": True})


@app.route("/api/videos/<path:video_key>/unskip", methods=["POST"])
@login_required
def unskip_video(video_key):
    username = session.get("username", "")
    conn = get_db()
    conn.execute(
        "DELETE FROM skipped_videos WHERE video_key = ? AND annotator = ?",
        (video_key, username)
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route("/api/videos/<path:video_key>/model_outputs")
@login_required
def get_model_outputs(video_key):
    outputs = load_model_outputs(video_key)
    chinese = load_chinese_outputs(video_key)
    return jsonify({
        "outputs": outputs or [],
        "chinese": chinese or {}
    })


@app.route("/api/videos/<path:video_key>/annotation", methods=["POST"])
@login_required
def save_annotation(video_key):
    data = request.json or {}
    if "user_id" not in session:
        return jsonify({"error": "not logged in"}), 401
    annotator = session["username"]

    fields = {
        "risk": data.get("risk", ""),
        "risk_subtype": data.get("risk_subtype", ""),
        "level1_scene": data.get("level1_scene", ""),
        "level2_subject": data.get("level2_subject", ""),
        "level3_risk_type": data.get("level3_risk_type", ""),
        "description": data.get("description", ""),
        "risk_localization": data.get("risk_localization", ""),
        "precursor_start_time": data.get("precursor_start_time", ""),
        "solution_for_person": data.get("solution_for_person", ""),
        "solution_for_hazard_source": data.get("solution_for_hazard_source", ""),
        "solution_prevent_recurrence": data.get("solution_prevent_recurrence", ""),
        "trim_segments": data.get("trim_segments", ""),
        "notes": data.get("notes", ""),
    }

    conn = get_db()
    c = conn.cursor()
    if not c.execute("SELECT 1 FROM videos WHERE video_key = ?", (video_key,)).fetchone():
        conn.close()
        return jsonify({"error": "Video not found"}), 404
    
    # 保存旧记录到历史表
    old = c.execute("SELECT * FROM human_annotations WHERE video_key = ? AND annotator = ?", (video_key, annotator)).fetchone()
    if old:
        c.execute("""
            INSERT INTO annotation_history (video_key, annotator, risk, risk_subtype, level1_scene, level2_subject, level3_risk_type,
                description, risk_localization, solution_for_person, solution_for_hazard_source,
                solution_prevent_recurrence, notes, trim_segments, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (old['video_key'], old['annotator'], old['risk'], old['risk_subtype'], old['level1_scene'], old['level2_subject'],
              old['level3_risk_type'], old['description'], old['risk_localization'], old['solution_for_person'],
              old['solution_for_hazard_source'], old['solution_prevent_recurrence'], old['notes'], old['trim_segments'], old['updated_at']))
    
    from datetime import datetime, timedelta, timezone
    bj_now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute(
        """
        INSERT INTO human_annotations (
            video_key, annotator, risk, risk_subtype, level1_scene, level2_subject, level3_risk_type,
            description, risk_localization, precursor_start_time, solution_for_person, solution_for_hazard_source,
            solution_prevent_recurrence, trim_segments, notes, updated_at, precursor_reviewed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_key, annotator) DO UPDATE SET
            risk=excluded.risk,
            risk_subtype=excluded.risk_subtype,
            level1_scene=excluded.level1_scene,
            level2_subject=excluded.level2_subject,
            level3_risk_type=excluded.level3_risk_type,
            description=excluded.description,
            risk_localization=excluded.risk_localization,
            precursor_start_time=excluded.precursor_start_time,
            solution_for_person=excluded.solution_for_person,
            solution_for_hazard_source=excluded.solution_for_hazard_source,
            solution_prevent_recurrence=excluded.solution_prevent_recurrence,
            trim_segments=excluded.trim_segments,
            notes=excluded.notes,
            updated_at=excluded.updated_at,
            precursor_reviewed_at=excluded.precursor_reviewed_at
        """,
        (video_key, annotator, *fields.values(), bj_now, bj_now),
    )
    # If the user saved an abnormal video with an empty precursor time,
    # treat it as "skip" so the video moves to the reviewed bucket.
    if (not fields["precursor_start_time"].strip()
            and fields["risk"] == "Yes"
            and fields["risk_subtype"] == "abnormal"):
        c.execute(
            "INSERT OR IGNORE INTO skipped_videos (video_key, annotator) VALUES (?, ?)",
            (video_key, annotator),
        )
    conn.commit()
    conn.close()
    broadcast_refresh()
    return jsonify({"success": True})


@app.route("/api/videos/<path:video_key>/media")
@login_required
def serve_video(video_key):
    if video_key in video_path_cache:
        path = resolve_data_path(video_path_cache[video_key])
    else:
        conn = get_db()
        row = conn.execute("SELECT file_path FROM videos WHERE video_key = ?", (video_key,)).fetchone()
        conn.close()
        if not row:
            return jsonify({"error": "Video not found in database"}), 404
        path = resolve_data_path(row["file_path"])
    video_path_cache[video_key] = path
    
    # 检查路径是否在允许的目录下
    allowed = False
    for allowed_root in [PUBLIC_DATA_ROOT.resolve(), GEN_DATA_ROOT.resolve()]:
        try:
            path.relative_to(allowed_root)
            allowed = True
            break
        except ValueError:
            continue
    
    if not allowed:
        app.logger.error(f"Video path {path} is outside allowed roots: PUBLIC_DATA_ROOT={PUBLIC_DATA_ROOT.resolve()}, GEN_DATA_ROOT={GEN_DATA_ROOT.resolve()}")
        return jsonify({"error": "Video path is outside allowed roots"}), 403
    
    if not path.exists():
        return jsonify({"error": f"Missing video file: {path}"}), 404
    response = send_file(path, conditional=True)
    response.headers['Cache-Control'] = 'public, max-age=86400'
    return response


@app.route("/api/trimmed_media/<path:file_path>")
@login_required
def serve_trimmed_video(file_path):
    path = (BATCH_TRIM_OUTPUT / file_path).resolve()
    try:
        path.relative_to(BATCH_TRIM_OUTPUT.resolve())
    except ValueError:
        return jsonify({"error": "Invalid path"}), 403
    if not path.exists():
        return jsonify({"error": "Trimmed video not found"}), 404
    response = send_file(path, conditional=True)
    response.headers['Cache-Control'] = 'public, max-age=86400'
    return response


@app.route("/api/sync_generated_videos", methods=["POST"])
@login_required
def sync_generated_videos():
    """Sync generated video results from batch outputs."""
    try:
        from import_generated_videos import import_generated_videos
        db_path = APP_DIR / "data" / "app.db"
        stats = import_generated_videos(db_path)
        broadcast_refresh()
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/datasets")
@login_required
def get_datasets():
    conn = get_db()
    rows = conn.execute(
        """
        SELECT v.dataset, COUNT(*) AS count
        FROM videos v
        GROUP BY v.dataset ORDER BY v.dataset
        """
    ).fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])


@app.route("/api/stats")
@login_required
def get_stats():
    username = session.get("username", "")
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM model_predictions").fetchone()[0]
    risk_stats = {
        row["risk"]: row["count"]
        for row in conn.execute("SELECT risk, COUNT(*) AS count FROM model_predictions GROUP BY risk")
    }

    # Global: videos annotated by anyone
    global_annotated = conn.execute(
        "SELECT COUNT(DISTINCT video_key) FROM human_annotations"
    ).fetchone()[0]
    global_pending = total - global_annotated

    # Per-user stats
    user_annotated = 0
    if username:
        user_annotated = conn.execute("""
            SELECT COUNT(*) FROM (
              SELECT video_key FROM human_annotations WHERE annotator = ?
              UNION
              SELECT video_key FROM skipped_videos WHERE annotator = ?
            )
        """, (username, username)).fetchone()[0]

    # Global progress (all annotators)
    # For unannotated videos, use model prediction; for annotated videos, use human annotation
    progress = {
        "risk": {"total": 0, "annotated": 0},
        "normal": {"total": 0, "annotated": 0},
    }
    
    # Get counts of annotated videos by their human annotation risk value
    annotated_risk_counts = {}
    for row in conn.execute(
        """
        SELECT a.risk, COUNT(DISTINCT a.video_key) AS count
        FROM human_annotations a
        GROUP BY a.risk
        """
    ):
        annotated_risk_counts[row["risk"]] = row["count"]
        if row["risk"] == "Yes":
            progress["risk"]["annotated"] = row["count"]
        elif row["risk"] == "No":
            progress["normal"]["annotated"] = row["count"]
    
    # Count unannotated videos by model prediction
    for row in conn.execute(
        """
        SELECT p.risk, COUNT(*) AS count
        FROM model_predictions p
        WHERE p.video_key NOT IN (SELECT DISTINCT video_key FROM human_annotations)
        GROUP BY p.risk
        """
    ):
        if row["risk"] == "Yes":
            progress["risk"]["total"] = annotated_risk_counts.get("Yes", 0) + row["count"]
        elif row["risk"] == "No":
            progress["normal"]["total"] = annotated_risk_counts.get("No", 0) + row["count"]
    
    # If no unannotated videos of a type, still set total from annotated
    if progress["risk"]["total"] == 0:
        progress["risk"]["total"] = annotated_risk_counts.get("Yes", 0)
    if progress["normal"]["total"] == 0:
        progress["normal"]["total"] = annotated_risk_counts.get("No", 0)

    # Risk subtype breakdown (异常行为 vs 风险前兆)
    annotated_subtype_counts = {
        row["risk_subtype"]: row["count"]
        for row in conn.execute(
            "SELECT a.risk_subtype, COUNT(DISTINCT a.video_key) AS count "
            "FROM human_annotations a WHERE a.risk = 'Yes' GROUP BY a.risk_subtype"
        )
    }
    annotated_abnormal = annotated_subtype_counts.get("abnormal", 0)
    annotated_risk_only = annotated_subtype_counts.get("risk_only", 0)
    annotated_risk_total = annotated_risk_counts.get("Yes", 0)
    total_risk = progress["risk"]["total"]

    if annotated_risk_total > 0:
        ratio_abnormal = annotated_abnormal / annotated_risk_total
    else:
        ratio_abnormal = 0.5

    unannotated_risk = max(0, total_risk - annotated_risk_total)
    estimated_abnormal_total = annotated_abnormal + round(unannotated_risk * ratio_abnormal)
    estimated_risk_only_total = total_risk - estimated_abnormal_total

    progress["risk_abnormal"] = {"total": estimated_abnormal_total, "annotated": annotated_abnormal}
    progress["risk_risk_only"] = {"total": estimated_risk_only_total, "annotated": annotated_risk_only}

    # Split progress: generated_videos vs others
    def get_split_progress(is_generated):
        if is_generated:
            where = "v.dataset = 'generated_videos'"
        else:
            where = "v.dataset != 'generated_videos'"
        
        p = {"risk": {"total": 0, "annotated": 0}, "normal": {"total": 0, "annotated": 0},
             "risk_abnormal": {"total": 0, "annotated": 0}, "risk_risk_only": {"total": 0, "annotated": 0}}
        
        # For generated_videos, use video_key path to determine risk type
        if is_generated:
            # Count by video_key pattern
            for row in conn.execute(f"""
                SELECT 
                    CASE 
                        WHEN v.video_key LIKE '%/abnormal/%' THEN 'abnormal'
                        WHEN v.video_key LIKE '%/risk_only/%' THEN 'risk_only'
                        WHEN v.video_key LIKE '%/normal/%' THEN 'normal'
                        ELSE 'unknown'
                    END as risk_type,
                    COUNT(*) as cnt
                FROM videos v
                JOIN model_predictions p2 ON p2.video_key = v.video_key
                WHERE {where}
                GROUP BY risk_type
            """):
                rt = row["risk_type"]
                cnt = row["cnt"]
                if rt == "abnormal":
                    p["risk_abnormal"]["total"] = cnt
                    p["risk"]["total"] += cnt
                elif rt == "risk_only":
                    p["risk_risk_only"]["total"] = cnt
                    p["risk"]["total"] += cnt
                elif rt == "normal":
                    p["normal"]["total"] = cnt
            
            # Count annotated by video_key pattern
            for row in conn.execute(f"""
                SELECT 
                    CASE 
                        WHEN v.video_key LIKE '%/abnormal/%' THEN 'abnormal'
                        WHEN v.video_key LIKE '%/risk_only/%' THEN 'risk_only'
                        WHEN v.video_key LIKE '%/normal/%' THEN 'normal'
                        ELSE 'unknown'
                    END as risk_type,
                    COUNT(DISTINCT a.video_key) as cnt
                FROM human_annotations a
                JOIN videos v ON v.video_key = a.video_key
                WHERE {where}
                GROUP BY risk_type
            """):
                rt = row["risk_type"]
                cnt = row["cnt"]
                if rt == "abnormal":
                    p["risk_abnormal"]["annotated"] = cnt
                    p["risk"]["annotated"] += cnt
                elif rt == "risk_only":
                    p["risk_risk_only"]["annotated"] = cnt
                    p["risk"]["annotated"] += cnt
                elif rt == "normal":
                    p["normal"]["annotated"] = cnt
            
            # Count keep (risk=Yes) by video_key pattern for keep ratio
            keep_counts = {}
            for row in conn.execute(f"""
                SELECT 
                    CASE 
                        WHEN v.video_key LIKE '%/abnormal/%' THEN 'abnormal'
                        WHEN v.video_key LIKE '%/risk_only/%' THEN 'risk_only'
                        WHEN v.video_key LIKE '%/normal/%' THEN 'normal'
                        ELSE 'unknown'
                    END as risk_type,
                    COUNT(DISTINCT a.video_key) as cnt
                FROM human_annotations a
                JOIN videos v ON v.video_key = a.video_key
                WHERE {where} AND a.risk = 'Yes'
                GROUP BY risk_type
            """):
                keep_counts[row["risk_type"]] = row["cnt"]
            
            p["keep_ratio"] = {
                "risk": keep_counts.get("abnormal", 0) + keep_counts.get("risk_only", 0),
                "risk_abnormal": keep_counts.get("abnormal", 0),
                "risk_risk_only": keep_counts.get("risk_only", 0),
                "normal": keep_counts.get("normal", 0),
            }
        else:
            # For non-generated videos, classify by the LATEST annotation per video.
            ann_risk_counts = {}
            for row in conn.execute(f"""
                SELECT a.risk, COUNT(DISTINCT v.video_key) AS count
                FROM videos v
                JOIN human_annotations a ON a.video_key = v.video_key
                WHERE {where}
                  AND a.updated_at = (
                    SELECT MAX(a2.updated_at) FROM human_annotations a2
                    WHERE a2.video_key = a.video_key
                  )
                  AND NOT (
                    v.video_key LIKE 'web_h264/%'
                    AND EXISTS (SELECT 1 FROM human_annotations ha WHERE ha.video_key = v.video_key AND ha.trim_segments IS NOT NULL AND ha.trim_segments != '')
                    AND EXISTS (SELECT 1 FROM videos seg WHERE seg.video_key LIKE REPLACE(REPLACE(v.video_key, 'web_h264/', 'web_h264_segments/'), '.mp4', '') || '_%')
                  )
                GROUP BY a.risk
            """):
                ann_risk_counts[row["risk"]] = row["count"]
                if row["risk"] == "Yes":
                    p["risk"]["annotated"] = row["count"]
                elif row["risk"] == "No":
                    p["normal"]["annotated"] = row["count"]
            
            for row in conn.execute(f"""
                SELECT p2.risk, COUNT(*) AS count
                FROM model_predictions p2 JOIN videos v ON v.video_key = p2.video_key
                WHERE {where} AND p2.video_key NOT IN (SELECT DISTINCT video_key FROM human_annotations)
                GROUP BY p2.risk
            """):
                if row["risk"] == "Yes":
                    p["risk"]["total"] = ann_risk_counts.get("Yes", 0) + row["count"]
                elif row["risk"] == "No":
                    p["normal"]["total"] = ann_risk_counts.get("No", 0) + row["count"]
            
            if p["risk"]["total"] == 0:
                p["risk"]["total"] = ann_risk_counts.get("Yes", 0)
            if p["normal"]["total"] == 0:
                p["normal"]["total"] = ann_risk_counts.get("No", 0)
            
            sub_counts = {}
            for row in conn.execute(f"""
                SELECT a.risk_subtype, COUNT(DISTINCT v.video_key) AS count
                FROM videos v
                JOIN human_annotations a ON a.video_key = v.video_key
                WHERE {where} AND a.risk = 'Yes' AND a.risk_subtype IS NOT NULL AND a.risk_subtype != ''
                  AND a.updated_at = (
                    SELECT MAX(a2.updated_at) FROM human_annotations a2
                    WHERE a2.video_key = a.video_key
                  )
                  AND NOT (
                    v.video_key LIKE 'web_h264/%'
                    AND EXISTS (SELECT 1 FROM human_annotations ha WHERE ha.video_key = v.video_key AND ha.trim_segments IS NOT NULL AND ha.trim_segments != '')
                    AND EXISTS (SELECT 1 FROM videos seg WHERE seg.video_key LIKE REPLACE(REPLACE(v.video_key, 'web_h264/', 'web_h264_segments/'), '.mp4', '') || '_%')
                  )
                GROUP BY a.risk_subtype
            """):
                sub_counts[row["risk_subtype"]] = row["count"]

            # The progress panel should reflect the sample-counting dataset
            # convention: latest real-video labels are counted directly, and
            # precursor segment samples are also represented as abnormal.
            display_sub_counts = {}
            for row in conn.execute(f"""
                SELECT a.risk_subtype, COUNT(DISTINCT v.video_key) AS count
                FROM videos v
                JOIN human_annotations a ON a.video_key = v.video_key
                WHERE {where} AND a.risk = 'Yes' AND a.risk_subtype IS NOT NULL AND a.risk_subtype != ''
                  AND a.updated_at = (
                    SELECT MAX(a2.updated_at) FROM human_annotations a2
                    WHERE a2.video_key = a.video_key
                  )
                GROUP BY a.risk_subtype
            """):
                display_sub_counts[row["risk_subtype"]] = row["count"]
            
            ann_ab = display_sub_counts.get("abnormal", sub_counts.get("abnormal", 0))
            ann_ro = display_sub_counts.get("risk_only", sub_counts.get("risk_only", 0))
            ann_risk = ann_ab + ann_ro
            tot_risk = p["risk"]["total"]
            
            if ann_risk > 0:
                ratio = ann_ab / ann_risk
            else:
                ratio = 0.5
            
            unann = max(0, tot_risk - ann_risk)
            p["risk"]["total"] = max(tot_risk, ann_risk)
            p["risk"]["annotated"] = max(p["risk"]["annotated"], ann_risk)
            p["risk_abnormal"] = {"total": ann_ab, "annotated": ann_ab}
            p["risk_risk_only"] = {"total": ann_ro, "annotated": ann_ro}
            
            # "异常转前兆": latest annotation is abnormal AND some
            # annotation has precursor_start_time filled.
            # Exclude web_h264 originals that have trim_segments +
            # corresponding segments (same logic as the video list).
            ab_transition = conn.execute(f"""
                SELECT COUNT(DISTINCT v.video_key) AS count
                FROM videos v
                JOIN human_annotations a ON a.video_key = v.video_key
                WHERE {where} AND a.risk = 'Yes' AND a.risk_subtype = 'abnormal'
                  AND a.updated_at = (
                    SELECT MAX(a2.updated_at) FROM human_annotations a2
                    WHERE a2.video_key = a.video_key
                  )
                  AND EXISTS (
                    SELECT 1 FROM human_annotations a3
                    WHERE a3.video_key = v.video_key
                    AND a3.precursor_start_time IS NOT NULL AND a3.precursor_start_time != ''
                  )
                  AND NOT (
                    v.video_key LIKE 'web_h264/%'
                    AND EXISTS (SELECT 1 FROM human_annotations ha WHERE ha.video_key = v.video_key AND ha.trim_segments IS NOT NULL AND ha.trim_segments != '')
                    AND EXISTS (SELECT 1 FROM videos seg WHERE seg.video_key LIKE REPLACE(REPLACE(v.video_key, 'web_h264/', 'web_h264_segments/'), '.mp4', '') || '_%')
                  )
            """).fetchone()
            risk_segment_samples = count_risk_segment_samples()
            p["abnormal_transition"] = {
                "annotated": risk_segment_samples or (ab_transition["count"] if ab_transition else 0)
            }
            p["risk"]["total"] += risk_segment_samples
            p["risk"]["annotated"] += risk_segment_samples
            p["risk_abnormal"]["total"] += risk_segment_samples
            p["risk_abnormal"]["annotated"] += risk_segment_samples

            # Count abnormal_revisited: latest annotation is abnormal AND
            # (some annotation has precursor_start_time OR any user skipped).
            # Progress is global (shared across all users).
            # Also exclude web_h264 originals that have trim_segments +
            # corresponding segments (same logic as the video list).
            ab_revisited = conn.execute(f"""
                SELECT COUNT(DISTINCT v.video_key) AS count
                FROM videos v
                JOIN human_annotations a ON a.video_key = v.video_key
                WHERE {where} AND a.risk = 'Yes' AND a.risk_subtype = 'abnormal'
                  AND a.updated_at = (
                    SELECT MAX(a2.updated_at) FROM human_annotations a2
                    WHERE a2.video_key = a.video_key
                  )
                  AND (
                    EXISTS (
                      SELECT 1 FROM human_annotations a3
                      WHERE a3.video_key = v.video_key
                        AND a3.precursor_start_time IS NOT NULL
                        AND a3.precursor_start_time != ''
                    )
                    OR EXISTS (
                      SELECT 1 FROM skipped_videos s
                      WHERE s.video_key = v.video_key
                    )
                  )
                  AND NOT (
                    v.video_key LIKE 'web_h264/%'
                    AND EXISTS (SELECT 1 FROM human_annotations ha WHERE ha.video_key = v.video_key AND ha.trim_segments IS NOT NULL AND ha.trim_segments != '')
                    AND EXISTS (SELECT 1 FROM videos seg WHERE seg.video_key LIKE REPLACE(REPLACE(v.video_key, 'web_h264/', 'web_h264_segments/'), '.mp4', '') || '_%')
                  )
            """).fetchone()
            p["abnormal_revisited"] = {
                "annotated": max(ab_revisited["count"] if ab_revisited else 0, risk_segment_samples)
            }

            # Total abnormal videos: latest annotation is abnormal.
            # Also exclude web_h264 originals that have trim_segments +
            # corresponding segments (same logic as the video list).
            ab_total = conn.execute(f"""
                SELECT COUNT(DISTINCT v.video_key) AS count
                FROM videos v
                JOIN human_annotations a ON a.video_key = v.video_key
                WHERE {where} AND a.risk = 'Yes' AND a.risk_subtype = 'abnormal'
                  AND a.updated_at = (
                    SELECT MAX(a2.updated_at) FROM human_annotations a2
                    WHERE a2.video_key = a.video_key
                  )
                  AND NOT (
                    v.video_key LIKE 'web_h264/%'
                    AND EXISTS (
                      SELECT 1 FROM human_annotations ha
                      WHERE ha.video_key = v.video_key
                      AND ha.trim_segments IS NOT NULL AND ha.trim_segments != ''
                    )
                    AND EXISTS (
                      SELECT 1 FROM videos seg
                      WHERE seg.video_key LIKE REPLACE(REPLACE(v.video_key, 'web_h264/', 'web_h264_segments/'), '.mp4', '') || '_%'
                    )
                  )
            """).fetchone()
            p["abnormal_total"] = ann_ab + risk_segment_samples
        
        return p

    progress_generated = get_split_progress(True)
    progress_other = get_split_progress(False)

    conn.close()
    return jsonify({
        "total": total,
        "annotated": global_annotated,
        "pending": global_pending,
        "user_annotated": user_annotated,
        "risk": risk_stats,
        "progress": progress,
        "progress_generated": progress_generated,
        "progress_other": progress_other
    })


@app.route("/api/stats/distribution")
@login_required
def get_stats_distribution():
    conn = get_db()
    c = conn.cursor()
    datasets = [row["dataset"] for row in c.execute("SELECT DISTINCT dataset FROM videos ORDER BY dataset")]
    human_fields = ["risk", "risk_subtype", "level1_scene", "level2_subject", "level3_risk_type"]
    model_fields = ["risk", "level1_scene", "level2_subject", "level3_risk_type"]
    result = {"datasets": datasets, "fields": {}}
    for field in human_fields:
        human_data = {}
        model_data = {}
        all_values = set()
        for ds in datasets:
            rows = c.execute(f"""
                SELECT a.{field}, COUNT(*) AS count
                FROM human_annotations a
                JOIN videos v ON v.video_key = a.video_key
                WHERE v.dataset = ? AND a.{field} IS NOT NULL AND a.{field} != ''
                GROUP BY a.{field} ORDER BY count DESC
            """, (ds,)).fetchall()
            vals = {row[field]: row["count"] for row in rows}
            human_data[ds] = vals
            all_values.update(vals.keys())

            if field in model_fields:
                rows = c.execute(f"""
                    SELECT p.{field}, COUNT(*) AS count
                    FROM model_predictions p
                    JOIN videos v ON v.video_key = p.video_key
                    WHERE v.dataset = ? AND p.{field} IS NOT NULL AND p.{field} != ''
                    GROUP BY p.{field} ORDER BY count DESC
                """, (ds,)).fetchall()
                vals = {row[field]: row["count"] for row in rows}
                model_data[ds] = vals
                all_values.update(vals.keys())

        sorted_labels = sorted(all_values, key=lambda x: str(x))
        result["fields"][field] = {
            "labels": sorted_labels,
            "human": human_data,
            "model": model_data,
        }
    conn.close()
    return jsonify(result)


@app.route("/api/device_stats")
@login_required
def get_device_stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM model_predictions").fetchone()[0]
    rows = conn.execute(
        """
        SELECT display_name, COUNT(DISTINCT video_key) AS count FROM (
          SELECT COALESCE(u.username, a.annotator) AS display_name, a.video_key
          FROM human_annotations a
          LEFT JOIN users u ON u.username = a.annotator
          UNION
          SELECT COALESCE(u2.username, s.annotator) AS display_name, s.video_key
          FROM skipped_videos s
          LEFT JOIN users u2 ON u2.username = s.annotator
        )
        GROUP BY display_name
        ORDER BY count DESC
        """
    ).fetchall()
    # Hide the internal "system" placeholder (used during data import).
    rows = [r for r in rows if r["display_name"] != "system"]
    conn.close()
    devices = [{"annotator": row["display_name"], "count": row["count"]} for row in rows]
    return jsonify({"total": total, "devices": devices})


@app.route("/api/export")
@login_required
def export_annotations():
    username = session.get("username", "")
    conn = get_db()
    query = """
        SELECT v.*, p.risk AS model_risk, p.level1_scene AS model_level1_scene,
               p.level2_subject AS model_level2_subject, p.level3_risk_type AS model_level3_risk_type,
               p.description AS model_description, p.risk_localization AS model_risk_localization,
               p.solution_for_person AS model_solution_for_person,
               p.solution_for_hazard_source AS model_solution_for_hazard_source,
               p.solution_prevent_recurrence AS model_solution_prevent_recurrence,
               a.annotator, a.risk, a.risk_subtype, a.level1_scene, a.level2_subject, a.level3_risk_type,
               a.description, a.risk_localization, a.solution_for_person,
               a.solution_for_hazard_source, a.solution_prevent_recurrence, a.trim_segments, a.notes, a.updated_at
        FROM videos v
        JOIN model_predictions p ON p.video_key = v.video_key
        LEFT JOIN human_annotations a ON a.video_key = v.video_key
        WHERE a.annotator = ?
    """
    rows = [dict(row) for row in conn.execute(query, (username,)).fetchall()]
    conn.close()
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    return jsonify({"filename": "human_annotations.jsonl", "content": content})


def parse_trim_segments(trim_str: str) -> list[tuple[float, float]]:
    segments = []
    # 统一替换中文符号，去掉方括号
    trim_str = trim_str.replace('，', ',').replace('；', ';').replace('[', '').replace(']', '').rstrip(';').rstrip('；').rstrip(',')
    for seg in trim_str.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        parts = seg.split(",")
        if len(parts) == 2:
            try:
                start = float(parts[0].strip())
                end = float(parts[1].strip())
                if start < end:
                    segments.append((start, end))
            except ValueError:
                continue
    return segments


def _trimmed_filename(stem: str, segments: list[tuple[float, float]], suffix: str) -> str:
    if len(segments) == 1:
        start, end = segments[0]
        return f"{stem}_trimmed_{start:.1f}-{end:.1f}{suffix}"
    segment_names = "_".join(f"{s:.1f}-{e:.1f}" for s, e in segments)
    candidate = f"{stem}_trimmed_{segment_names}{suffix}"
    if len(candidate.encode('utf-8')) > 200:
        h = hashlib.md5(segment_names.encode()).hexdigest()[:12]
        return f"{stem}_trimmed_{len(segments)}seg_{h}{suffix}"
    return candidate


@app.route("/api/videos/<path:video_key>/trim", methods=["POST"])
@login_required
def trim_video(video_key):
    data = request.json or {}
    trim_segments_str = data.get("trim_segments", "").strip()
    if not trim_segments_str:
        return jsonify({"error": "trim_segments is required"}), 400

    segments = parse_trim_segments(trim_segments_str)
    if not segments:
        return jsonify({"error": "No valid segments found. Use format: start,end;start,end"}), 400

    conn = get_db()
    row = conn.execute("SELECT file_path FROM videos WHERE video_key = ?", (video_key,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Video not found"}), 404

    video_path = Path(row["file_path"]).resolve()
    try:
        video_path.relative_to(PUBLIC_DATA_ROOT.resolve())
    except ValueError:
        return jsonify({"error": "Video path is outside public data root"}), 403
    if not video_path.exists():
        return jsonify({"error": f"Missing video file: {video_path}"}), 404

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    suffix = video_path.suffix
    stem = video_path.stem

    if len(segments) == 1:
        start, end = segments[0]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cmd = [
                ffmpeg_exe, "-y", "-ss", str(start),
                "-i", str(video_path),
                "-to", str(end - start),
                "-c", "copy", "-avoid_negative_ts", "make_zero",
                tmp_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            output_name = _trimmed_filename(stem, segments, suffix)
            return send_file(
                tmp_path,
                as_attachment=True,
                download_name=output_name,
                mimetype="video/mp4"
            )
        except subprocess.CalledProcessError as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return jsonify({"error": f"FFmpeg error: {e.stderr.decode()[-500:]}"}), 500
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            concat_list = Path(tmp_dir) / "concat.txt"
            part_files = []
            for i, (start, end) in enumerate(segments):
                part_path = Path(tmp_dir) / f"part{i}{suffix}"
                cmd = [
                    ffmpeg_exe, "-y", "-ss", str(start),
                    "-i", str(video_path),
                    "-to", str(end - start),
                    "-c", "copy", "-avoid_negative_ts", "make_zero",
                    str(part_path)
                ]
                subprocess.run(cmd, capture_output=True, check=True)
                part_files.append(part_path)

            with open(concat_list, "w") as f:
                for p in part_files:
                    f.write(f"file '{p}'\n")

            output_path = Path(tmp_dir) / f"output{suffix}"
            cmd = [
                ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c", "copy", str(output_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            output_name = _trimmed_filename(stem, segments, suffix)
            return send_file(
                str(output_path),
                as_attachment=True,
                download_name=output_name,
                mimetype="video/mp4"
            )


BATCH_TRIM_OUTPUT = Path("/home/caiqingyuan/code/lifebench/data/public_data/seg_video")


@app.route("/api/batch_trim", methods=["POST"])
@login_required
def batch_trim():
    annotator = request.args.get("annotator", "").strip()
    dataset = request.args.get("dataset", "").strip()
    BATCH_TRIM_OUTPUT.mkdir(parents=True, exist_ok=True)

    conn = get_db()
    query = """
        SELECT v.video_key, v.file_path, a.annotator, a.trim_segments
        FROM human_annotations a
        JOIN videos v ON v.video_key = a.video_key
        WHERE a.trim_segments IS NOT NULL AND a.trim_segments != ''
    """
    params = []
    if annotator:
        query += " AND a.annotator = ?"
        params.append(annotator)
    if dataset:
        query += " AND v.dataset = ?"
        params.append(dataset)

    rows = conn.execute(query, params).fetchall()

    if not rows:
        conn.close()
        return jsonify({"error": "没有找到需要裁切的视频", "results": []}), 400

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    results = []

    for row in rows:
        video_key = row["video_key"]
        file_path = Path(row["file_path"]).resolve()
        trim_segments_str = row["trim_segments"]

        try:
            file_path.relative_to(PUBLIC_DATA_ROOT.resolve())
        except ValueError:
            results.append({"video_key": video_key, "status": "skipped", "reason": "路径不在公开目录"})
            continue

        if not file_path.exists():
            results.append({"video_key": video_key, "status": "skipped", "reason": "文件不存在"})
            continue

        segments = parse_trim_segments(trim_segments_str)
        if not segments:
            results.append({"video_key": video_key, "status": "skipped", "reason": "无有效裁切段"})
            continue

        suffix = file_path.suffix
        stem = file_path.stem
        relative_path = file_path.relative_to(PUBLIC_DATA_ROOT.resolve())
        output_subdir = BATCH_TRIM_OUTPUT / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        try:
            if len(segments) == 1:
                start, end = segments[0]
                output_path = output_subdir / _trimmed_filename(stem, segments, suffix)
                cmd = [
                    ffmpeg_exe, "-y", "-ss", str(start),
                    "-i", str(file_path),
                    "-to", str(end - start),
                    "-c", "copy", "-avoid_negative_ts", "make_zero",
                    str(output_path)
                ]
                subprocess.run(cmd, capture_output=True, check=True)
            else:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    concat_list = Path(tmp_dir) / "concat.txt"
                    part_files = []
                    for i, (start, end) in enumerate(segments):
                        part_path = Path(tmp_dir) / f"part{i}{suffix}"
                        cmd = [
                            ffmpeg_exe, "-y", "-ss", str(start),
                            "-i", str(file_path),
                            "-to", str(end - start),
                            "-c", "copy", "-avoid_negative_ts", "make_zero",
                            str(part_path)
                        ]
                        subprocess.run(cmd, capture_output=True, check=True)
                        part_files.append(part_path)

                    with open(concat_list, "w") as f:
                        for p in part_files:
                            f.write(f"file '{p}'\n")

                    output_path = output_subdir / _trimmed_filename(stem, segments, suffix)
                    cmd = [
                        ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
                        "-i", str(concat_list),
                        "-c", "copy", str(output_path)
                    ]
                    subprocess.run(cmd, capture_output=True, check=True)

            # 标记为已裁切
            conn.execute(
                "UPDATE human_annotations SET trimmed = 1 WHERE video_key = ? AND annotator = ?",
                (video_key, row["annotator"])
            )
            conn.commit()

            results.append({
                "video_key": video_key,
                "status": "success",
                "output": str(output_path.relative_to(BATCH_TRIM_OUTPUT)),
                "segments": trim_segments_str
            })
        except subprocess.CalledProcessError as e:
            results.append({
                "video_key": video_key,
                "status": "error",
                "reason": e.stderr.decode()[-200:] if e.stderr else "ffmpeg error"
            })

    conn.close()
    success_count = sum(1 for r in results if r["status"] == "success")
    return jsonify({
        "output_dir": str(BATCH_TRIM_OUTPUT),
        "total": len(rows),
        "success": success_count,
        "failed": len(rows) - success_count,
        "results": results
    })


@app.route("/api/git_backup", methods=["POST"])
@login_required
def git_backup():
    username = session.get("username", "")
    if username not in ("ligaoxiang", "caiqingyuan"):
        return jsonify({"error": "无权限"}), 403

    import datetime
    import gzip

    APP_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = APP_DIR.parents[1]
    DB_PATH = APP_DIR / "data" / "app.db"
    SQL_GZ_PATH = APP_DIR / "data" / "app.db.sql.gz"
    LAST_BACKUP_FILE = APP_DIR / "data" / "last_backup.txt"

    conn = get_db()
    current_count = conn.execute("SELECT COUNT(*) FROM human_annotations").fetchone()[0]
    conn.close()

    last_count = 0
    if LAST_BACKUP_FILE.exists():
        try:
            last_count = int(LAST_BACKUP_FILE.read_text().strip())
        except:
            last_count = 0

    new_annotations = current_count - last_count

    subprocess.run(["rm", "-f", str(SQL_GZ_PATH)], capture_output=True)
    result = subprocess.run(
        ["sqlite3", str(DB_PATH), ".dump"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return jsonify({"error": f"SQLite导出失败: {result.stderr}"}), 500

    import io
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(result.stdout.encode("utf-8"))
    with open(SQL_GZ_PATH, "wb") as f:
        f.write(buf.getvalue())

    LAST_BACKUP_FILE.write_text(str(current_count))

    subprocess.run(["git", "-C", str(PROJECT_ROOT), "add", "annotation/public_video_annot/data/app.db.sql.gz", ".gitignore"], capture_output=True)
    commit_msg = f"Backup {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | +{new_annotations} annotations | total: {current_count}"
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "commit", "-m", commit_msg],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return jsonify({"error": f"Git commit失败: {result.stderr}"}), 500

    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "push", "origin", "lifebench"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return jsonify({"error": f"Git push失败: {result.stderr}"}), 500

    return jsonify({"success": True, "message": commit_msg})


if __name__ == "__main__":
    init_db()
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", "5002")), threaded=True)
