import json
import os
import sqlite3
import subprocess
import tempfile
import hashlib
import secrets
from pathlib import Path
from functools import wraps
import time

import requests
from flask import Flask, jsonify, request, send_file, send_from_directory, session, redirect, url_for, Response
import imageio_ffmpeg

clients = []
online_users = set()
last_update = {"video_key": None, "timestamp": 0}


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
PUBLIC_DATA_ROOT = PROJECT_ROOT / "data" / "public_data"
MODEL_OUTPUTS_ROOTS = [
    PROJECT_ROOT / "public_data_filter" / "batch_outputs" / "persistent_full" / "reviews",
    PROJECT_ROOT / "public_data_filter" / "batch_outputs" / "web_20260429T170230Z" / "reviews",
]

VOLCENGINE_TRANSLATE_CONFIG = {
    "access_key_id": os.environ.get("VOLCENGINE_ACCESS_KEY", "AKLTZjBlZjE3Nzc3MmU5NGRhYzlkOGUwNGE3OTViODMxZGM"),
    "secret_access_key": os.environ.get("VOLCENGINE_SECRET_KEY", "TW1ZeE1ETXpOelUxTUROak5ETTROV0ZpWVdJNU9UVXdZVEkyWmpKaU4yRQ=="),
}

trans_cache = {}


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
        "None",
    ],
}

app = Flask(__name__)
app.config["DATABASE"] = str(APP_DIR / "data" / "app.db")
app.secret_key = secrets.token_hex(32)


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
    conn = sqlite3.connect(app.config["DATABASE"])
    conn.row_factory = sqlite3.Row
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
    return send_from_directory(APP_DIR / "static", "index.html")


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


def broadcast_refresh():
    global last_update
    last_update = {"timestamp": time.time()}


@app.route("/api/videos")
@login_required
def get_videos():
    dataset = request.args.get("dataset", "")
    status = request.args.get("status", "")
    username = session.get("username", "")
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
               p.majority_json, p.agents_json, p.summary_json
        FROM videos v
        JOIN model_predictions p ON p.video_key = v.video_key
        LEFT JOIN model_prediction_zh z ON z.video_key = v.video_key
        WHERE 1=1
    """
    params = []
    if dataset:
        query += " AND v.dataset = ?"
        params.append(dataset)
    if status == "risk":
        query += " AND p.risk = 'Yes'"
    elif status == "normal":
        query += " AND p.risk = 'No'"
    elif status == "problem":
        query += " AND p.yes_count > 0 AND p.no_count > 0"
    elif status == "pending":
        query += " AND v.video_key NOT IN (SELECT video_key FROM human_annotations)"
        query += " ORDER BY COALESCE(p.yes_count, 0) DESC, COALESCE(p.votes_cast, 0) DESC, v.dataset, v.video_key LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    elif status == "annotated":
        query += " AND EXISTS (SELECT 1 FROM human_annotations h WHERE h.video_key = v.video_key)"
        query += " ORDER BY (SELECT MAX(h.updated_at) FROM human_annotations h WHERE h.video_key = v.video_key) DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    else:
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
        row["media_url"] = f"/api/videos/{row['video_key']}/media"
        row["majority_vote"] = json.loads(row.pop("majority_json") or "{}")
        row["qwen_summary_meta"] = json.loads(row.pop("agents_json") or "{}")
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

    conn.close()
    return jsonify(rows)


@app.route("/api/taxonomy")
@login_required
def get_taxonomy():
    return jsonify(TAXONOMY)


def load_translated_output(video_key):
    """Load pre-translated model outputs for a video."""
    translated_base = PROJECT_ROOT / "public_data_filter" / "batch_outputs" / "translated"
    
    for dataset_dir in translated_base.iterdir():
        if not dataset_dir.is_dir():
            continue
        if video_key.startswith(dataset_dir.name + '/'):
            rest_path = video_key[len(dataset_dir.name)+1:]
            parts = rest_path.split('/')
            # Strip extension from filename
            parts[-1] = Path(parts[-1]).stem
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
    for reviews_dir in MODEL_OUTPUTS_ROOTS:
        if not reviews_dir.exists():
            continue
        for dataset_dir in reviews_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            if not video_key.startswith(dataset_dir.name):
                continue
            rest_path = video_key[len(dataset_dir.name)+1:]
            parts = rest_path.split('/')
            parts[-1] = Path(parts[-1]).stem
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
                outputs = []
                for agent_idx, agent in enumerate(data.get("agent_results", [])):
                    pd = agent.get("parsed_decision", {})
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
                    })
                return outputs
            except Exception as e:
                print(f"Error loading model outputs for {video_key}: {e}")
                pass
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
    
    c.execute(
        """
        INSERT INTO human_annotations (
            video_key, annotator, risk, risk_subtype, level1_scene, level2_subject, level3_risk_type,
            description, risk_localization, solution_for_person, solution_for_hazard_source,
            solution_prevent_recurrence, trim_segments, notes, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(video_key, annotator) DO UPDATE SET
            risk=excluded.risk,
            risk_subtype=excluded.risk_subtype,
            level1_scene=excluded.level1_scene,
            level2_subject=excluded.level2_subject,
            level3_risk_type=excluded.level3_risk_type,
            description=excluded.description,
            risk_localization=excluded.risk_localization,
            solution_for_person=excluded.solution_for_person,
            solution_for_hazard_source=excluded.solution_for_hazard_source,
            solution_prevent_recurrence=excluded.solution_prevent_recurrence,
            trim_segments=excluded.trim_segments,
            notes=excluded.notes,
            updated_at=datetime('now')
        """,
        (video_key, annotator, *fields.values()),
    )
    conn.commit()
    conn.close()
    broadcast_refresh()
    return jsonify({"success": True})


@app.route("/api/videos/<path:video_key>/media")
@login_required
def serve_video(video_key):
    conn = get_db()
    row = conn.execute("SELECT file_path FROM videos WHERE video_key = ?", (video_key,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Video not found in database"}), 404
    path = Path(row["file_path"]).resolve()
    try:
        path.relative_to(PUBLIC_DATA_ROOT.resolve())
    except ValueError:
        return jsonify({"error": "Video path is outside public data root"}), 403
    if not path.exists():
        return jsonify({"error": f"Missing video file: {path}"}), 404
    return send_file(path, conditional=True)


@app.route("/api/datasets")
@login_required
def get_datasets():
    conn = get_db()
    rows = conn.execute(
        """
        SELECT v.dataset, COUNT(*) AS count
        FROM videos v JOIN model_predictions p ON p.video_key = v.video_key
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
        user_annotated = conn.execute(
            "SELECT COUNT(*) FROM human_annotations WHERE annotator = ?", (username,)
        ).fetchone()[0]

    # Global progress (all annotators)
    progress = {
        "risk": {"total": risk_stats.get("Yes", 0), "annotated": 0},
        "normal": {"total": risk_stats.get("No", 0), "annotated": 0},
    }
    for row in conn.execute(
        """
        SELECT p.risk, COUNT(DISTINCT a.video_key) AS count
        FROM human_annotations a
        JOIN model_predictions p ON p.video_key = a.video_key
        GROUP BY p.risk
        """
    ):
        if row["risk"] == "Yes":
            progress["risk"]["annotated"] = row["count"]
        elif row["risk"] == "No":
            progress["normal"]["annotated"] = row["count"]
    conn.close()
    return jsonify({
        "total": total,
        "annotated": global_annotated,
        "pending": global_pending,
        "user_annotated": user_annotated,
        "risk": risk_stats,
        "progress": progress
    })


@app.route("/api/device_stats")
@login_required
def get_device_stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM model_predictions").fetchone()[0]
    rows = conn.execute(
        """
        SELECT COALESCE(u.username, a.annotator) AS display_name, COUNT(*) AS count
        FROM human_annotations a
        LEFT JOIN users u ON u.username = a.annotator
        GROUP BY display_name
        ORDER BY count DESC
        """
    ).fetchall()
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
                ffmpeg_exe, "-y", "-i", str(video_path),
                "-ss", str(start), "-to", str(end),
                "-c", "copy", "-avoid_negative_ts", "make_zero",
                tmp_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            output_name = f"{stem}_trimmed_{start:.1f}-{end:.1f}{suffix}"
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
                    ffmpeg_exe, "-y", "-i", str(video_path),
                    "-ss", str(start), "-to", str(end),
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

            segment_names = "_".join(f"{s:.1f}-{e:.1f}" for s, e in segments)
            output_name = f"{stem}_trimmed_{segment_names}{suffix}"
            return send_file(
                str(output_path),
                as_attachment=True,
                download_name=output_name,
                mimetype="video/mp4"
            )


BATCH_TRIM_OUTPUT = Path("/data_4/liuyuan/lifebench/data/public_data/seg_video")


@app.route("/api/batch_trim", methods=["POST"])
@login_required
def batch_trim():
    annotator = request.args.get("annotator", "").strip()
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

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
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
                output_path = output_subdir / f"{stem}_trimmed_{start:.1f}-{end:.1f}{suffix}"
                cmd = [
                    ffmpeg_exe, "-y", "-i", str(file_path),
                    "-ss", str(start), "-to", str(end),
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
                            ffmpeg_exe, "-y", "-i", str(file_path),
                            "-ss", str(start), "-to", str(end),
                            "-c", "copy", "-avoid_negative_ts", "make_zero",
                            str(part_path)
                        ]
                        subprocess.run(cmd, capture_output=True, check=True)
                        part_files.append(part_path)

                    with open(concat_list, "w") as f:
                        for p in part_files:
                            f.write(f"file '{p}'\n")

                    segment_names = "_".join(f"{s:.1f}-{e:.1f}" for s, e in segments)
                    output_path = output_subdir / f"{stem}_trimmed_{segment_names}{suffix}"
                    cmd = [
                        ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
                        "-i", str(concat_list),
                        "-c", "copy", str(output_path)
                    ]
                    subprocess.run(cmd, capture_output=True, check=True)

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

    success_count = sum(1 for r in results if r["status"] == "success")
    return jsonify({
        "output_dir": str(BATCH_TRIM_OUTPUT),
        "total": len(rows),
        "success": success_count,
        "failed": len(rows) - success_count,
        "results": results
    })


if __name__ == "__main__":
    init_db()
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", "5002")))
