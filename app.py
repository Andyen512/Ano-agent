#!/usr/bin/env python3
"""
通用视频标注框架 - Flask 应用

基于步骤的视频标注系统，支持:
- 用户认证
- 视频列表和筛选
- 分步标注 (可配置步骤)
- 批量裁切
- 导出标注数据

使用方法:
1. 配置 config.py
2. 运行: python app.py
"""
import os
import json
import sqlite3
import hashlib
import secrets
import tempfile
import subprocess
from pathlib import Path
from functools import wraps

import imageio_ffmpeg
from flask import Flask, jsonify, request, send_file, send_from_directory, session, redirect, url_for, Response

# 导入配置
from config import (
    SECRET_KEY, DATABASE_PATH, PORT, PUBLIC_DATA_ROOT,
    resolve_video_path, DATABASE_TABLES, TAXONOMY, ANNOTATION_STEPS,
    DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD, SITE_TITLE, to_chinese,
)

app = Flask(__name__)
app.config["DATABASE"] = str(DATABASE_PATH)
app.secret_key = SECRET_KEY

# 确保目录存在
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated


def get_db():
    conn = sqlite3.connect(app.config["DATABASE"])
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column(conn, table: str, column: str, definition: str) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db():
    """初始化数据库表"""
    conn = get_db()
    c = conn.cursor()
    
    for table_name, table_def in DATABASE_TABLES.items():
        c.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({table_def})")
    
    # 创建用户表
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 创建默认管理员
    if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        c.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (DEFAULT_ADMIN_USERNAME, hash_password(DEFAULT_ADMIN_PASSWORD))
        )
    
    conn.commit()
    conn.close()


# ================== 认证路由 ==================

@app.route("/login", methods=["GET"])
def login_page():
    return send_from_directory(app.root_path + "/static", "login.html")


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


# ================== 用户管理 ==================

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


# ================== 主页面 ==================

@app.route("/")
@login_required
def index():
    return send_from_directory(app.root_path + "/static", "index.html")


# ================== 视频API ==================

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
               p.description AS model_description, p.risk_localization AS model_risk_localization,
               p.solution_for_person AS model_solution_for_person,
               p.solution_for_hazard_source AS model_solution_for_hazard_source,
               p.solution_prevent_recurrence AS model_solution_prevent_recurrence,
               p.yes_count, p.no_count, p.votes_cast, p.complete,
               p.majority_json, p.agents_json, p.summary_json
        FROM videos v
        JOIN model_predictions p ON p.video_key = v.video_key
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
        all_anns = c.execute(
            "SELECT * FROM human_annotations WHERE video_key = ? ORDER BY updated_at DESC",
            (row["video_key"],),
        ).fetchall()
        row["annotations"] = [dict(a) for a in all_anns]
        row["annotation"] = dict(all_anns[0]) if all_anns else None
        if username:
            for a in all_anns:
                if a["annotator"] == username:
                    row["annotation"] = dict(a)
                    break

    conn.close()
    return jsonify(rows)


@app.route("/api/videos/<path:video_key>/media")
@login_required
def serve_video(video_key):
    conn = get_db()
    row = conn.execute("SELECT file_path FROM videos WHERE video_key = ?", (video_key,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Video not found"}), 404
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


@app.route("/api/taxonomy")
@login_required
def get_taxonomy():
    return jsonify(TAXONOMY)


# ================== 标注保存 ==================

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
        "notes": data.get("notes", ""),
        "trim_segments": data.get("trim_segments", ""),
    }

    conn = get_db()
    c = conn.cursor()
    if not c.execute("SELECT 1 FROM videos WHERE video_key = ?", (video_key,)).fetchone():
        conn.close()
        return jsonify({"error": "Video not found"}), 404

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
    return jsonify({"success": True})


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


# ================== 统计 ==================

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
    global_annotated = conn.execute(
        "SELECT COUNT(DISTINCT video_key) FROM human_annotations"
    ).fetchone()[0]
    global_pending = total - global_annotated
    user_annotated = 0
    if username:
        user_annotated = conn.execute(
            "SELECT COUNT(*) FROM human_annotations WHERE annotator = ?", (username,)
        ).fetchone()[0]
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


# ================== 导出 ==================

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


# ================== 视频裁切 ==================

def parse_trim_segments(trim_str: str) -> list:
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


if __name__ == "__main__":
    init_db()
    app.run(debug=False, host="0.0.0.0", port=PORT)
