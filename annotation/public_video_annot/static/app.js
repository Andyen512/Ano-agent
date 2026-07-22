const state = { videos: [], totalCount: 0, current: null, taxonomy: {}, stats: null, deviceStats: null, currentUsername: null, latestRequestId: 0, currentStep: 0, modelOutputs: null, chineseOutputs: null, allDatasets: [], revisit: "unreviewed", searchTerm: "", searchTimer: null };

const $ = (id) => document.getElementById(id);

function autoResizeTextarea(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = textarea.scrollHeight + 'px';
}

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
  if (e.code === "Space") {
    e.preventDefault();
    const video = document.querySelector("#detail video");
    if (video) {
      if (video.paused) video.play(); else video.pause();
    }
  }
  if (e.code === "ArrowLeft") {
    e.preventDefault();
    const prev = document.getElementById("prev-video");
    if (prev && !prev.disabled) prev.click();
  }
  if (e.code === "ArrowRight") {
    e.preventDefault();
    const next = document.getElementById("next-video");
    if (next && !next.disabled) next.click();
  }
});
document.addEventListener("keyup", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
  if (e.code === "Space") e.preventDefault();
});

function getOrCreateDeviceId() {
  let id = localStorage.getItem("lifebench_device_id");
  if (!id) {
    const bytes = new Uint8Array(4);
    crypto.getRandomValues(bytes);
    const suffix = [...bytes].map(b => b.toString(16).padStart(2, "0")).join("");
    id = `device_${suffix}`;
    localStorage.setItem("lifebench_device_id", id);
  }
  return id;
}

function annotator() {
  return state.currentUsername || getOrCreateDeviceId();
}

function modelValue(video, key) {
  return video[`model_${key}`] || "";
}

async function loadDatasets() {
  const res = await fetch("/api/datasets");
  state.allDatasets = await res.json();
  filterAndRenderDatasets();
}

function filterAndRenderDatasets() {
  const videoType = $("video-type").value;
  let datasets = state.allDatasets;
  if (videoType === "generated") {
    datasets = datasets.filter(d => d.dataset === "generated_videos");
  } else if (videoType === "real") {
    datasets = datasets.filter(d => d.dataset !== "generated_videos");
  }
  const prevDataset = $("dataset").value;
  $("dataset").innerHTML = `<option value="">全部数据集</option>` + datasets.map(d => `<option value="${d.dataset}">${d.dataset} (${d.count})</option>`).join("");
  if (prevDataset && datasets.some(d => d.dataset === prevDataset)) {
    $("dataset").value = prevDataset;
  } else if (datasets.length === 1) {
    $("dataset").value = datasets[0].dataset;
  }
  resizeDatasetSelect();
}

function resizeDatasetSelect() {
  const sel = $("dataset");
  const opt = sel.options[sel.selectedIndex];
  if (!opt) return;
  const tmp = document.createElement("span");
  tmp.style.cssText = "visibility:hidden;position:absolute;white-space:nowrap;font:inherit";
  tmp.textContent = opt.text;
  document.body.appendChild(tmp);
  sel.style.width = (tmp.offsetWidth + 30) + "px";
  document.body.removeChild(tmp);
}

async function loadTaxonomy() {
  const res = await fetch("/api/taxonomy");
  state.taxonomy = await res.json();
}

async function loadStats() {
  const res = await fetch(`/api/stats?annotator=${encodeURIComponent(annotator())}`);
  const s = await res.json();
  state.stats = s;
  $("stats").textContent = `总数 ${s.total} | 已标注 ${s.annotated || 0} | 待标 ${s.pending || s.total}`;
}

async function loadDeviceStats() {
  const res = await fetch("/api/device_stats");
  state.deviceStats = await res.json();
  // Re-render progress panel if a video is selected
  if (state.current) {
    const progressEl = document.querySelector('.progress-panel');
    if (progressEl) {
      progressEl.outerHTML = progressPanel();
    }
  }
}

async function loadVideos(append = false) {
  append = append === true;
  const requestId = ++state.latestRequestId;
  const prevLength = state.videos.length;
  const dataset = $("dataset").value;
  const sortBy = dataset === "generated_videos" ? $("sort-by").value : "";
  const revisit = $("video-type").value === "real"
    && $("status").value === "annotated"
    && $("risk-subtype-filter").value === "abnormal"
    ? (state.revisit || "")
    : "";
  const params = new URLSearchParams({
    annotator: annotator(),
    dataset: dataset,
    video_type: $("video-type").value,
    risk_subtype: $("risk-subtype-filter").value,
    status: $("status").value,
    duration: $("duration-filter").value,
    sort_by: sortBy,
    revisit: revisit,
    search: state.searchTerm,
    offset: append ? state.videos.length : 0,
    limit: "50",
  });
  const res = await fetch(`/api/videos?${params}`);
  if (requestId !== state.latestRequestId) return;
  const data = await res.json();
  if (requestId !== state.latestRequestId) return;
  const newVideos = data.videos || [];
  state.totalCount = data.total || 0;
  if (append) {
    state.videos.push(...newVideos);
  } else {
    state.videos = newVideos;
  }
  renderList(append, prevLength);
  await Promise.all([loadStats(), loadDeviceStats()]);
}

async function reloadAndSelect(index) {
  await loadVideos();
  if (!state.videos.length) {
    state.current = null;
    $("detail").innerHTML = `<div class="empty">当前筛选下没有待标注视频</div>`;
    return;
  }
  selectVideo(Math.max(0, Math.min(index, state.videos.length - 1)));
}

function renderList(append = false, prevLength = 0) {
  const startIdx = append ? prevLength : 0;
  const isGeneratedVideos = $("dataset").value === "generated_videos";
  const html = state.videos.slice(startIdx).map((v, i) => {
    const idx = startIdx + i;
    const riskClass = v.model_risk === "Yes" ? "risk-yes" : "risk-no";
    const hasTrim = v.annotation?.trim_segments;
    const isSkipped = v.is_skipped;
    const annotator = v.annotation?.annotator;
    const viewingUsers = v.viewing_users || [];
    const viewingIndicator = viewingUsers.length > 0
      ? `<span class="viewing-indicator" title="正在标注: ${viewingUsers.join(', ')}" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;margin-left:4px;"></span>`
      : '';
    
    // 生成视频显示分数和风险类别
    let scoreHtml = "";
    let riskBadge = `<span class="badge ${riskClass}">模型: ${v.model_risk}</span>`;
    
    if (isGeneratedVideos) {
      // 从视频路径提取 abnormal/risk_only 类别
      // 路径格式: generated_videos/场景/主体/风险类型/描述/模型/abnormal或risk_only/视频.mp4
      const parts = v.video_key.split('/');
      if (parts.length >= 7) {
        const videoType = parts[6];  // abnormal 或 risk_only
        const typeClass = videoType === "abnormal" ? "risk-yes" : "risk-no";
        riskBadge = `<span class="badge ${typeClass}">${videoType}</span>`;
      }
      
      if (v.majority_vote?.avg_scores) {
        const avgScore = v.majority_vote.avg_scores.avg_total_score;
        if (avgScore !== null && avgScore !== undefined) {
          scoreHtml = `<span class="badge" style="background:#6366f1">分数: ${avgScore.toFixed(1)}</span>`;
        }
      }
    }
    
    return `<div class="item ${state.current?.video_key === v.video_key ? "active" : ""}" data-i="${idx}">
      <strong>${v.video_key}${viewingIndicator}</strong>
      <div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:2px;">
        ${riskBadge}
        ${scoreHtml}
        ${v.annotation ? `<span class="badge">已标${annotator ? '(' + annotator + ')' : ''}</span>` : ""}
        ${v.annotation?.precursor_start_time ? `<span class="badge" style="background:#8b5cf6">前兆时刻${v.annotation.precursor_start_time}s</span>` : ""}
        ${isSkipped ? `<span class="badge" style="background:#f59e0b">已跳过</span>` : ""}
        ${hasTrim ? `<span class="badge trim-badge">需裁切</span>` : ""}
      </div>
    </div>`;
  }).join("");

  const precursorAnnotated = state.stats?.progress_other?.abnormal_revisited?.annotated
    ?? state.stats?.progress_other?.abnormal_transition?.annotated
    ?? 0;
  const precursorTotal = state.stats?.progress_other?.abnormal_total ?? state.totalCount;
  const showPrecursorProgress = $("video-type").value === "real"
    && $("status").value === "annotated"
    && $("risk-subtype-filter").value === "abnormal";
  const precursorHtml = showPrecursorProgress
    ? (() => {
        const unreviewedCount = Math.max(0, precursorTotal - precursorAnnotated);
        const baseStyle = "margin-left:4px;padding:1px 6px;font-size:11px;border:1px solid #8b5cf6;border-radius:3px;cursor:pointer;";
        const activeStyle = "background:#8b5cf6;color:#fff;";
        const inactiveStyle = "background:#fff;color:#8b5cf6;";
        const isUnreviewed = state.revisit === "unreviewed";
        const isReviewed = state.revisit === "reviewed";
        const allStyle = !isUnreviewed && !isReviewed ? activeStyle : inactiveStyle;
        return `<div class="list-count" style="margin-top:2px;color:#8b5cf6;display:flex;align-items:center;flex-wrap:wrap;gap:2px;">
          <span>前兆已标 ${precursorAnnotated}/${precursorTotal}</span>
          <button data-revisit="" style="${baseStyle}${allStyle}">全部</button>
          <button data-revisit="unreviewed" style="${baseStyle}${isUnreviewed ? activeStyle : inactiveStyle}">未标 (${unreviewedCount})</button>
          <button data-revisit="reviewed" style="${baseStyle}${isReviewed ? activeStyle : inactiveStyle}">已标 (${precursorAnnotated})</button>
        </div>`;
      })()
    : "";
  const countHtml = `<div class="list-count">共 ${state.totalCount} 个视频</div>${precursorHtml}`;
  
  if (append) {
    $("load-more")?.insertAdjacentHTML("beforebegin", html);
    if (state.videos.length === prevLength) {
      const btn = $("load-more");
      if (btn) btn.style.display = "none";
    }
  } else {
    $("list").innerHTML = countHtml + html + `<button id="load-more" class="secondary" style="width:100%;margin-top:8px">加载更多</button>`;
    $("load-more").onclick = () => loadVideos(true);
  }
  [...document.querySelectorAll("#list [data-revisit]")].forEach(btn => {
    btn.onclick = (e) => {
      e.stopPropagation();
      const next = btn.dataset.revisit;
      if (state.revisit === next) return;
      state.revisit = next;
      loadVideos();
    };
  });
  [...document.querySelectorAll(".item")].forEach(el => {
    el.onclick = (e) => {
      const selection = window.getSelection();
      if (selection.toString().length > 0) return;
      selectVideo(Number(el.dataset.i));
    };
  });
}

const LABEL_MAP = {
  risk: { Yes: "有风险", No: "无风险" },
  risk_subtype: { abnormal: "异常行为", risk_only: "仅有异常前兆" },
  level1_scene: { "dining room": "餐厅", "kitchen": "厨房", "study": "书房", "balcony": "阳台", "living room": "客厅", "bathroom": "浴室", "bedroom": "卧室", "yard": "院子" },
  level2_subject: { child: "儿童", "older adult": "老年人", "young adult": "年轻人", "middle-aged adult": "中年人", all: "所有人" },
  level3_risk_type: { "fall/instability": "跌倒/不稳", "heat/fire source": "热源/火灾", "collision/crush injury": "碰撞/挤压伤", "sharp-object danger": "锐器危险", "electrical safety": "电气安全", "poisoning/accidental ingestion": "中毒/误食", "interpersonal conflict": "人际冲突", "animal attack/biosecurity risk": "动物攻击/生物安全风险", "stranger theft": "陌生人偷盗", "None": "无" },
};

function toChinese(key, val) {
  return LABEL_MAP[key]?.[val] || val || "-";
}

function field(name, label, value, wide = false, textarea = false) {
  const cls = wide ? "wide" : "";
  if (textarea) {
    return `<label class="${cls}">${label}<textarea id="${name}">${value || ""}</textarea></label>`;
  }
  return `<label class="${cls}">${label}<input id="${name}" value="${value || ""}" /></label>`;
}

function selectField(name, label, value, options) {
  const current = value || "";
  const allOptions = options.includes(current) || !current ? options : [current, ...options];
  return `<label>${label}<select id="${name}">
    ${allOptions.map(opt => `<option value="${opt}" ${opt === current ? "selected" : ""}>${toChinese(name, opt)}</option>`).join("")}
  </select></label>`;
}

function progressBar(label, progress, color) {
  const total = progress?.total || 0;
  const annotated = progress?.annotated || 0;
  const pct = total > 0 ? Math.round((annotated / total) * 100) : 0;
  const fillStyle = color ? `style="width: ${pct}%; background: ${color}"` : `style="width: ${pct}%"`;
  if (label) {
    return `<div class="progress-row">
      <div class="progress-label"><span>${label}</span><span>${annotated}/${total} (${pct}%)</span></div>
      <div class="progress-track"><div class="progress-fill" ${fillStyle}></div></div>
    </div>`;
  }
  return `<div class="progress-mini">
    <div class="progress-label"><span>${annotated}/${total} (${pct}%)</span></div>
    <div class="progress-track"><div class="progress-fill" ${fillStyle}></div></div>
  </div>`;
}

function progressPanel() {
  const pg = state.stats?.progress_generated || {};
  const po = state.stats?.progress_other || {};
  const devices = state.deviceStats?.devices || [];
  const kr = pg.keep_ratio || {};
  
  function progressBarMini(label, progress, color, keepText) {
    const total = progress?.total || 0;
    const annotated = progress?.annotated || 0;
    const pct = total > 0 ? Math.round((annotated / total) * 100) : 0;
    const fillStyle = color ? `style="width: ${pct}%; background: ${color}"` : `style="width: ${pct}%"`;
    return `<div style="display:flex;flex-direction:column;gap:2px;">
      <div style="font-size:11px;color:#64748b;text-align:center;">${annotated}/${total} (${pct}%)${keepText || ''}</div>
      <div class="progress-track"><div class="progress-fill" ${fillStyle}></div></div>
    </div>`;
  }
  
  function progressBarStacked(baseProgress, extraCount, baseColor, extraColor) {
    const total = baseProgress?.total || 0;
    const baseAnnotated = baseProgress?.annotated || 0;
    const extraAnnotated = extraCount || 0;
    const pct = total > 0 ? Math.round((baseAnnotated / total) * 100) : 0;
    const extraPct = total > 0 ? Math.round((extraAnnotated / total) * 100) : 0;
    return `<div style="display:flex;flex-direction:column;gap:2px;">
      <div style="font-size:11px;color:#64748b;text-align:center;">${baseAnnotated}/${total} (${pct}%) <span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${extraColor};vertical-align:middle;"></span>异转前${extraAnnotated}</div>
      <div class="progress-track" style="display:flex;overflow:hidden;">
        <div class="progress-fill" style="width:${pct}%;background:${baseColor};min-width:0;"></div>
        <div class="progress-fill" style="width:${extraPct}%;background:${extraColor};min-width:0;"></div>
      </div>
    </div>`;
  }
  
  function keepRatioText(annotated, kept) {
    annotated = annotated || 0;
    kept = kept || 0;
    if (!annotated) return '';
    const pct = Math.round((kept / annotated) * 100);
    return ` <span style="color:#0f766e;font-weight:600;">保留${pct}%</span>`;
  }
  
  return `<div class="progress-panel">
    <div class="progress-row" style="font-weight:600;font-size:13px;color:#334155;">
      <div></div>
      <div style="text-align:center;">生成视频</div>
      <div style="text-align:center;">真实视频</div>
    </div>
    <div class="progress-row">
      <div class="progress-col-label">风险视频</div>
      <div class="progress-col">${progressBarMini("", pg.risk, null, keepRatioText(pg.risk?.annotated, kr.risk))}</div>
      <div class="progress-col">${progressBarMini("", po.risk)}</div>
    </div>
    <div class="progress-row">
      <div class="progress-col-label">  异常行为</div>
      <div class="progress-col">${progressBarMini("", pg.risk_abnormal, "#0d9488", keepRatioText(pg.risk_abnormal?.annotated, kr.risk_abnormal))}</div>
      <div class="progress-col">${progressBarMini("", po.risk_abnormal, "#0d9488")}</div>
    </div>
    <div class="progress-row">
      <div class="progress-col-label">  风险前兆</div>
      <div class="progress-col">${progressBarMini("", pg.risk_risk_only, "#f59e0b", keepRatioText(pg.risk_risk_only?.annotated, kr.risk_risk_only))}</div>
      <div class="progress-col">${progressBarStacked(po.risk_risk_only, po.abnormal_transition?.annotated, "#f59e0b", "#8b5cf6")}</div>
    </div>
    <div class="progress-row">
      <div class="progress-col-label">无风险视频</div>
      <div class="progress-col">${progressBarMini("", pg.normal, null, keepRatioText(pg.normal?.annotated, kr.normal))}</div>
      <div class="progress-col">${progressBarMini("", po.normal)}</div>
    </div>
    ${devices.length ? `<div class="device-stats-section">
      <div class="device-stats-title">设备标注数</div>
      ${devices.map(d => {
        const isMe = d.annotator === annotator();
        return `<div class="device-stats-row ${isMe ? "device-stats-me" : ""}">
          <span class="device-name">${isMe ? "本设备" : d.annotator}</span>
          <span class="device-count">${d.count}</span>
        </div>`;
      }).join('')}
    </div>` : ''}
  </div>`;
}

function renderAllAnnotations(annotations) {
  if (!annotations.length) return '';
  return `<div class="all-annotations">
    <h3>所有标注 (${annotations.length})</h3>
    <div class="annotations-list">
      ${annotations.map(a => `
        <div class="annotation-card">
          <div class="annotation-header">
            <span class="annotator">${a.annotator}</span>
            <span class="time">${a.updated_at || ''}</span>
          </div>
          <div class="annotation-body">
            <div><strong>Risk:</strong> ${toChinese('risk', a.risk)}</div>
            <div><strong>风险定位:</strong> ${a.risk_localization || '-'}</div>
            <div><strong>风险前兆开始时刻:</strong> ${a.precursor_start_time || '-'}</div>
            <div><strong>场景:</strong> ${toChinese('level1_scene', a.level1_scene)}</div>
            <div><strong>主体:</strong> ${toChinese('level2_subject', a.level2_subject)}</div>
            <div><strong>风险类型:</strong> ${toChinese('level3_risk_type', a.level3_risk_type)}</div>
            <div><strong>描述:</strong> ${a.description || '-'}</div>
          </div>
        </div>
      `).join('')}
    </div>
  </div>`;
}

function formatTime(t) {
  if (!t || isNaN(t) || !isFinite(t)) return '0:00.0';
  const m = Math.floor(t / 60);
  const s = t % 60;
  return `${m}:${s.toFixed(1).padStart(4, '0')}`;
}

function generateTimeline(duration) {
  if (!duration || duration <= 0) return { labels: '', ticks: '' };
  let labelInterval;
  if (duration <= 10) labelInterval = 1;
  else if (duration <= 30) labelInterval = 2;
  else if (duration <= 60) labelInterval = 5;
  else labelInterval = 10;
  const halfSteps = Math.ceil(duration / 0.5);
  let labels = '';
  let ticks = '';
  for (let i = 0; i <= halfSteps; i++) {
    const t = i * 0.5;
    const pct = (t / duration) * 100;
    ticks += `<div class="tick" style="left:${pct.toFixed(4)}%"></div>`;
    if (i % (labelInterval * 2) === 0) {
      labels += `<div class="tick-label" style="left:${pct.toFixed(4)}%">${formatTime(t)}</div>`;
    }
  }
  return { labels, ticks };
}

function bindVideoError(videoEl) {
  if (!videoEl) return;
  videoEl.addEventListener('error', () => {
    const errorEl = document.createElement('div');
    errorEl.className = 'video-error';
    errorEl.textContent = '视频加载失败';
    videoEl.replaceWith(errorEl);
  }, { once: true });
}

function selectVideo(index) {
  const v = state.videos[index];
  state.current = v;
  state.currentStep = 0;
  state.modelOutputs = null;
  state.chineseOutputs = null;
  const ann = v.annotation || {};
  const value = (key) => ann[key] ?? modelValue(v, key);
  const isGeneratedVideos = $("dataset").value === "generated_videos";
  
  navigator.sendBeacon("/api/current_viewing", new Blob([JSON.stringify({ video_key: v.video_key })], { type: "application/json" }));
  
  const videoSrc = v.trimmed_media_url || v.media_url;
  
  // 生成视频使用简化的标注界面
  if (isGeneratedVideos) {
    $("detail").innerHTML = `
      <div class="grid">
        <div>
          <div style="position:relative">
            <video autoplay loop muted preload="auto" src="${videoSrc}" style="width:100%"></video>
          </div>
          <div class="custom-video-controls" id="custom-controls">
            <div class="timeline-area">
              <div class="timeline-labels" id="timeline-labels"></div>
              <div class="timeline-wrapper">
                <div class="timeline-ticks" id="timeline-ticks"></div>
                <div class="timeline-track" id="timeline-track">
                  <div class="timeline-played" id="timeline-played" style="width:0%"></div>
                  <div class="timeline-thumb" id="timeline-thumb" style="left:0%"></div>
                </div>
              </div>
            </div>
            <div class="video-controls-row">
              <button id="play-btn" class="play-btn">▶</button>
              <span class="time-display" id="time-display">0:00.0 / 0:00.0</span>
              <div class="controls-spacer"></div>
              <input type="range" id="volume-slider" class="volume-slider" min="0" max="1" step="0.05" value="1" title="音量" />
              <button id="fullscreen-btn" class="play-btn" title="全屏">⛶</button>
            </div>
          </div>
          <div class="video-nav">
            <button id="prev-video" class="secondary" ${index <= 0 ? "disabled" : ""}>上一条</button>
            <button id="next-video" class="secondary" ${index >= state.videos.length - 1 ? "disabled" : ""}>下一条</button>
          </div>
          ${progressPanel()}
        </div>
        <div class="panel" style="display:flex;flex-direction:column;max-height:calc(100vh - 120px);">
          <div class="panel-header">
            <h2>人工标注</h2>
            <div class="header-nav">
              <button id="save" class="save-btn">保存</button>
              <button id="skip" class="step-btn secondary" style="color:#d97706">跳过</button>
            </div>
          </div>
          ${(() => {
            const parts = v.video_key.split('/');
            const videoType = parts.length >= 7 ? parts[6] : '';
            const typeLabel = videoType === 'abnormal' ? '异常行为' : videoType === 'risk_only' ? '风险前兆' : videoType;
            const typeColor = videoType === 'abnormal' ? '#dc2626' : videoType === 'risk_only' ? '#d97706' : '#6b7280';
            const typeBg = videoType === 'abnormal' ? '#fee2e2' : videoType === 'risk_only' ? '#fef3c7' : '#f1f5f9';
            const avgScore = v.majority_vote?.avg_scores?.avg_total_score;
            const scoreDisplay = avgScore !== null && avgScore !== undefined ? avgScore.toFixed(1) : 'N/A';
            return `<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:16px;flex:1;min-height:0;">
              <div style="display:flex;flex-direction:column;gap:14px;overflow:auto;">
                <div>
                  <div style="font-weight:600;margin-bottom:6px;font-size:13px;color:#334155;">原始 Prompt</div>
                  <div id="original-prompt-content" style="padding:12px;background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;font-size:13px;line-height:1.6;white-space:pre-wrap;max-height:260px;overflow:auto;">加载中...</div>
                </div>
                <div>
                  <div style="font-weight:600;margin-bottom:6px;font-size:13px;color:#334155;">中文翻译</div>
                  <div id="original-prompt-zh" style="padding:12px;background:#f0fdf4;border:1px solid #86efac;border-radius:8px;font-size:13px;line-height:1.6;white-space:pre-wrap;min-height:80px;max-height:260px;overflow:auto;">${state.modelOutputs?.[0]?.original_prompt_zh || "加载中..."}</div>
                </div>
              </div>
              <div style="display:flex;flex-direction:column;gap:16px;">
                <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
                  ${videoType ? `<span style="padding:6px 14px;background:${typeBg};border:1px solid ${typeColor};border-radius:6px;font-weight:600;color:${typeColor};font-size:14px;">${typeLabel}</span>` : ''}
                  <span style="padding:6px 14px;background:#eef2ff;border:1px solid #6366f1;border-radius:6px;font-weight:600;color:#4338ca;font-size:14px;">分数: ${scoreDisplay}</span>
                </div>
                <div>
                  <div style="font-weight:600;margin-bottom:10px;font-size:14px;color:#1e293b;">判定结果</div>
                  <div style="display:flex;gap:12px;">
                    <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
                      <input type="radio" name="keep_decision" value="keep" ${value("risk") === "Yes" ? "checked" : ""} style="width:18px;height:18px;">
                      <span style="padding:8px 16px;background:#dcfce7;border:1px solid #22c55e;border-radius:6px;font-weight:600;color:#166534;">保留</span>
                    </label>
                    <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
                      <input type="radio" name="keep_decision" value="drop" ${value("risk") === "No" ? "checked" : ""} style="width:18px;height:18px;">
                      <span style="padding:8px 16px;background:#fee2e2;border:1px solid #ef4444;border-radius:6px;font-weight:600;color:#991b1b;">不保留</span>
                    </label>
                  </div>
                </div>
              </div>
            </div>`;
          })()}
        </div>
      </div>`;
    
    $("save").onclick = saveCurrent;
    $("skip").onclick = skipCurrent;
    $("prev-video").onclick = () => moveSelection(-1);
    $("next-video").onclick = () => moveSelection(1);
    
    // 初始化视频控制
    const videoEl = document.querySelector('#detail video');
    if (videoEl) {
      bindVideoError(videoEl);
      const playBtn = $('play-btn');
      const timeDisplay = $('time-display');
      const timelinePlayed = $('timeline-played');
      const timelineThumb = $('timeline-thumb');
      const timelineTrack = $('timeline-track');
      const timelineLabels = $('timeline-labels');
      const timelineTicks = $('timeline-ticks');
      const volumeSlider = $('volume-slider');
      const fullscreenBtn = $('fullscreen-btn');

      videoEl.addEventListener('loadedmetadata', () => {
        const dur = videoEl.duration || 0;
        if (timeDisplay) timeDisplay.textContent = `0:00.0 / ${formatTime(dur)}`;
        if (timelineLabels && timelineTicks && dur > 0) {
          const { labels, ticks } = generateTimeline(dur);
          timelineLabels.innerHTML = labels;
          timelineTicks.innerHTML = ticks;
        }
      });

      videoEl.addEventListener('timeupdate', () => {
        const ct = videoEl.currentTime;
        const dur = videoEl.duration || 0;
        const pct = dur > 0 ? (ct / dur) * 100 : 0;
        if (timelinePlayed) timelinePlayed.style.width = pct + '%';
        if (timelineThumb) timelineThumb.style.left = pct + '%';
        if (timeDisplay) timeDisplay.textContent = `${formatTime(ct)} / ${formatTime(dur)}`;
      });

      if (playBtn) {
        playBtn.onclick = () => {
          if (videoEl.paused) { videoEl.play(); playBtn.textContent = '⏸'; }
          else { videoEl.pause(); playBtn.textContent = '▶'; }
        };
        videoEl.addEventListener('play', () => { playBtn.textContent = '⏸'; });
        videoEl.addEventListener('pause', () => { playBtn.textContent = '▶'; });
      }

      if (timelineTrack) {
        timelineTrack.addEventListener('click', (e) => {
          const rect = timelineTrack.getBoundingClientRect();
          const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
          if (videoEl.duration) videoEl.currentTime = pct * videoEl.duration;
        });
      }

      if (volumeSlider) {
        volumeSlider.addEventListener('input', () => {
          videoEl.volume = parseFloat(volumeSlider.value);
        });
      }

      if (fullscreenBtn) {
        fullscreenBtn.onclick = () => {
          if (videoEl.requestFullscreen) videoEl.requestFullscreen();
          else if (videoEl.webkitRequestFullscreen) videoEl.webkitRequestFullscreen();
        };
      }
    }
    
    loadModelOutputsForVideo(v.video_key);
    renderList();
    return;
  }
  
  $("detail").innerHTML = `
    <div class="grid">
      <div>
        <div style="position:relative">
          <video autoplay loop muted preload="auto" src="${videoSrc}" style="width:100%"></video>
          <div id="video-overlay" style="display:none;position:absolute;top:0;left:0;width:100%;height:100%;background:#1a1a1a;z-index:10"></div>
          <button id="toggle-video" style="position:absolute;top:8px;right:8px;z-index:20;padding:4px 8px;font-size:12px;background:rgba(0,0,0,0.6);color:#fff;border:1px solid rgba(255,255,255,0.3);border-radius:4px;cursor:pointer">隐藏画面</button>
        </div>
        <div class="custom-video-controls" id="custom-controls" style="display:none">
          <div class="timeline-area">
            <div class="timeline-labels" id="timeline-labels"></div>
            <div class="timeline-wrapper">
              <div class="timeline-ticks" id="timeline-ticks"></div>
              <div class="timeline-track" id="timeline-track">
                <div class="timeline-played" id="timeline-played" style="width:0%"></div>
                <div class="timeline-thumb" id="timeline-thumb" style="left:0%"></div>
              </div>
            </div>
          </div>
          <div class="video-controls-row">
            <button id="play-btn" class="play-btn">▶</button>
            <span class="time-display" id="time-display">0:00.0 / 0:00.0</span>
            <div class="controls-spacer"></div>
            <input type="range" id="volume-slider" class="volume-slider" min="0" max="1" step="0.05" value="1" title="音量" />
            <button id="fullscreen-btn" class="play-btn" title="全屏">⛶</button>
          </div>
        </div>
        ${v.trimmed_media_url ? '<div style="margin-top:4px;font-size:12px;color:#3b82f6">正在播放裁切后的视频</div>' : ''}
        <div class="video-nav">
          <button id="prev-video" class="secondary" ${index <= 0 ? "disabled" : ""}>上一条</button>
          <button id="next-video" class="secondary" ${index >= state.videos.length - 1 ? "disabled" : ""}>下一条</button>
        </div>
        ${progressPanel()}
        <div class="trim-section">
          <label class="trim-checkbox">
            <input type="checkbox" id="trim_needed" ${ann.trim_segments ? "checked" : ""} />
            <span>需要裁切</span>
          </label>
          <div id="trim-input-area" class="trim-input-area" style="display: ${ann.trim_segments ? "block" : "none"}">
            <label>裁切时间段
              <textarea id="trim_segments" placeholder="例如: 5,6;10,15">${ann.trim_segments || ""}</textarea>
            </label>
            <p class="trim-hint">格式: 开始秒,结束秒 多段用分号分隔 (如 5,6;10,15 表示裁切 5-6 秒和 10-15 秒)</p>
            <button id="trim-download" class="trim-download-btn">裁切并下载</button>
          </div>
        </div>
        <button id="show-prev-annotation" class="secondary" style="margin-top:8px">查看上一条标注</button>
        <div id="prev-annotation-content" style="display:none;margin-top:8px;padding:8px;background:#f5f5f5;border-radius:4px;font-size:12px;line-height:1.6"></div>
      </div>
      <div class="panel">
        <div class="panel-header">
          <h2>人工标注</h2>
          <div class="header-nav">
            <button id="prev-step" class="step-btn secondary" style="display:none">上一步</button>
            <button id="next-step" class="step-btn secondary">下一步</button>
            <button id="save" class="save-btn">保存</button>
            <button id="skip" class="step-btn secondary" style="color:#d97706">跳过</button>
            <button id="report-broken" class="step-btn secondary" style="color:#dc2626">视频无法加载</button>
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;padding:8px 16px;background:#fef3c7;border-bottom:1px solid #f59e0b;">
          <span style="font-size:13px;font-weight:600;color:#92400e;white-space:nowrap;">风险前兆开始时刻(秒):</span>
          <input id="precursor_start_time" value="${value("precursor_start_time") || ""}" style="flex:1;padding:6px 10px;border:1px solid #f59e0b;border-radius:4px;font-size:13px;" placeholder="如: 5（无则留空或点跳过）" />
          <button id="save-with-precursor" class="save-btn" style="padding:6px 14px;font-size:13px;">保存并下一条</button>
        </div>
        <div class="form">
          <div class="step-indicator" id="step-indicator"></div>
          <div id="step-form"></div>
          <div class="fixed-fields">
            <label>Risk
              <select id="risk">
                <option value="Yes" ${value("risk") === "Yes" ? "selected" : ""}>Yes</option>
                <option value="No" ${value("risk") === "No" ? "selected" : ""}>No</option>
              </select>
            </label>
            <div id="risk-fields">
              <label id="risk-subtype-label">风险类别
                <select id="risk_subtype">
                  <option value="abnormal" ${value("risk_subtype") === "abnormal" ? "selected" : ""}>${toChinese("risk_subtype","abnormal")}</option>
                  <option value="risk_only" ${value("risk_subtype") === "risk_only" ? "selected" : ""}>${toChinese("risk_subtype","risk_only")}</option>
                </select>
              </label>
              ${selectField("level1_scene", "Level 1 场景", value("level1_scene"), state.taxonomy.level1_scene || [])}
              ${selectField("level2_subject", "Level 2 主体", value("level2_subject"), state.taxonomy.level2_subject || [])}
              <label>Level 3 风险类型
                <select id="level3_risk_type">
                  ${(state.taxonomy.level3_risk_type || []).map(opt => `<option value="${opt}" ${value("level3_risk_type") === opt ? "selected" : ""}>${toChinese("level3_risk_type", opt)}</option>`).join("")}
                  <option value="__custom__" ${value("level3_risk_type") && !(state.taxonomy.level3_risk_type || []).includes(value("level3_risk_type")) ? "selected" : ""}>手动输入</option>
                </select>
              </label>
              <div id="level3_custom_area" style="display: ${value("level3_risk_type") && !(state.taxonomy.level3_risk_type || []).includes(value("level3_risk_type")) ? "block" : "none"}">
                <label>自定义风险类型<input id="level3_risk_type_custom" value="${value("level3_risk_type") || ""}" /></label>
              </div>
              <label class="inline-label"><span class="label-title">风险定位 <small>(如: 5,14)</small></span><input id="risk_localization" value="${value("risk_localization") || ""}" /></label>
            </div>
            <label class="inline-label">备注<textarea id="notes" class="small-textarea">${ann.notes || ""}</textarea></label>
          </div>
        </div>
      </div>
    </div>
    ${renderAllAnnotations(v.annotations || [])}`;
  
  $("save").onclick = saveCurrent;
  $("save-with-precursor").onclick = saveCurrent;
  $("skip").onclick = skipCurrent;
  $("prev-video").onclick = () => moveSelection(-1);
  $("next-video").onclick = () => moveSelection(1);
  $("prev-step").onclick = () => changeStep(-1);
  $("next-step").onclick = () => changeStep(1);
  $("report-broken").onclick = reportBrokenVideo;
  $("trim_needed").onchange = updateTrimVisibility;
  $("risk").onchange = () => { updateRiskFieldsVisibility(); renderStepForm(); };
  $("level3_risk_type").onchange = updateLevel3CustomVisibility;
  
  // 隐藏/显示视频画面
  $("toggle-video").onclick = () => {
    const overlay = $("video-overlay");
    const btn = $("toggle-video");
    if (overlay.style.display === "none") {
      overlay.style.display = "block";
      btn.textContent = "显示画面";
    } else {
      overlay.style.display = "none";
      btn.textContent = "隐藏画面";
    }
  };
  // Custom video player controls
  const videoEl = document.querySelector('#detail video');
  if (videoEl) {
    bindVideoError(videoEl);
    const playBtn = $('play-btn');
    const timeDisplay = $('time-display');
    const timelinePlayed = $('timeline-played');
    const timelineThumb = $('timeline-thumb');
    const timelineTrack = $('timeline-track');
    const timelineLabels = $('timeline-labels');
    const timelineTicks = $('timeline-ticks');
    const volumeSlider = $('volume-slider');
    const fullscreenBtn = $('fullscreen-btn');
    const customControls = $('custom-controls');

    videoEl.addEventListener('loadedmetadata', () => {
      const dur = videoEl.duration || 0;
      if (customControls) customControls.style.display = '';
      if (timeDisplay) timeDisplay.textContent = `0:00.0 / ${formatTime(dur)}`;
      if (timelineLabels && timelineTicks && dur > 0) {
        const { labels, ticks } = generateTimeline(dur);
        timelineLabels.innerHTML = labels;
        timelineTicks.innerHTML = ticks;
      }
    });

    videoEl.addEventListener('timeupdate', () => {
      const ct = videoEl.currentTime;
      const dur = videoEl.duration || 0;
      const pct = dur > 0 ? (ct / dur) * 100 : 0;
      if (timelinePlayed) timelinePlayed.style.width = pct + '%';
      if (timelineThumb) timelineThumb.style.left = pct + '%';
      if (timeDisplay) timeDisplay.textContent = `${formatTime(ct)} / ${formatTime(dur)}`;
    });

    if (playBtn) {
      playBtn.onclick = () => {
        if (videoEl.paused) { videoEl.play(); playBtn.textContent = '⏸'; }
        else { videoEl.pause(); playBtn.textContent = '▶'; }
      };
      videoEl.addEventListener('play', () => { playBtn.textContent = '⏸'; });
      videoEl.addEventListener('pause', () => { playBtn.textContent = '▶'; });
    }

    if (timelineTrack) {
      timelineTrack.addEventListener('click', (e) => {
        const rect = timelineTrack.getBoundingClientRect();
        const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        if (videoEl.duration) videoEl.currentTime = pct * videoEl.duration;
      });
    }

    if (volumeSlider) {
      volumeSlider.addEventListener('input', () => {
        videoEl.volume = parseFloat(volumeSlider.value);
      });
    }

    if (fullscreenBtn) {
      fullscreenBtn.onclick = () => {
        if (videoEl.requestFullscreen) videoEl.requestFullscreen();
        else if (videoEl.webkitRequestFullscreen) videoEl.webkitRequestFullscreen();
      };
    }

    if (videoEl.readyState >= 1) {
      videoEl.dispatchEvent(new Event('loadedmetadata'));
    }
  }
  updateRiskFieldsVisibility();
  updateLevel3CustomVisibility();

  // Toggle skip/unskip button based on current state
  const skipBtn = $("skip");
  if (v.is_skipped) {
    skipBtn.textContent = "取消跳过";
    skipBtn.onclick = unskipCurrent;
  } else {
    skipBtn.textContent = "跳过";
    skipBtn.onclick = skipCurrent;
  }

  loadModelOutputsForVideo(v.video_key);
  renderStepForm();
  renderList();
}

async function loadModelOutputsForVideo(videoKey) {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);
    const res = await fetch(`/api/videos/${encodeURIComponent(videoKey)}/model_outputs`, { signal: controller.signal });
    clearTimeout(timeoutId);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.modelOutputs = data.outputs || [];
    state.chineseOutputs = data.chinese || {};
    
    // 更新原始 prompt 显示
    const originalPromptEl = document.getElementById("original-prompt-content");
    if (originalPromptEl && state.modelOutputs[0]?.original_prompt) {
      originalPromptEl.textContent = state.modelOutputs[0].original_prompt;
    }
    // 更新中文翻译区域
    const promptZhEl = document.getElementById("original-prompt-zh");
    if (promptZhEl) {
      promptZhEl.textContent = state.modelOutputs[0]?.original_prompt_zh || "暂无翻译";
    }
  } catch (e) {
    console.error("Failed to load model outputs:", e);
    state.modelOutputs = [];
  } finally {
    renderStepForm();
  }
}

const STEPS = ["description", "solution_for_person", "solution_for_hazard_source", "solution_prevent_recurrence"];
const STEP_LABELS = { description: "正常视频描述或风险描述", solution_for_person: "解决方案-对人", solution_for_hazard_source: "解决方案-对危险源", solution_prevent_recurrence: "解决方案-防止危险复发" };

function renderStepForm() {
  if (!state.current) return;
  const isGeneratedVideos = $("dataset").value === "generated_videos";
  if (isGeneratedVideos) return;
  
  const step = STEPS[state.currentStep];
  const ann = state.current.annotation || {};
  const isRisk = $("risk")?.value === "Yes";
  const visibleSteps = isRisk ? STEPS : ["description"];
  if (!visibleSteps.includes(step)) {
    state.currentStep = 0;
  }
  
  // 如果是 description 步骤且没有标注，使用原始 prompt 作为默认值
  let currentValue = ann[step] || "";
  if (step === "description" && !currentValue && state.modelOutputs?.[0]?.original_prompt) {
    currentValue = state.modelOutputs[0].original_prompt;
  }

  let indicatorHtml = `<div class="step-dots">`;
  visibleSteps.forEach((s, i) => {
    const originalIndex = STEPS.indexOf(s);
    indicatorHtml += `<span class="step-dot ${s === step ? 'active' : ''} ${ann[s] ? 'completed' : ''}" data-step="${originalIndex}" onclick="jumpToStep(${originalIndex})">${STEP_LABELS[s].slice(0, 4)}</span>`;
  });
  indicatorHtml += `</div>`;
  if ($("step-indicator")) {
    $("step-indicator").innerHTML = indicatorHtml;
  }

  let modelOutputsHtml = `<div class="model-outputs">`;

  const stepFields = {
    "description": "description",
    "solution_for_person": "solution_for_person",
    "solution_for_hazard_source": "solution_for_hazard_source",
    "solution_prevent_recurrence": "solution_to_prevent_recurrence"
  };
  const zhFields = {
    "description": "description_zh",
    "solution_for_person": "solution_for_person_zh",
    "solution_for_hazard_source": "solution_for_hazard_source_zh",
    "solution_prevent_recurrence": "solution_to_prevent_recurrence_zh"
  };

  if (state.modelOutputs === null) {
    modelOutputsHtml += `<div class="model-loading">加载中...</div>`;
  } else if (state.modelOutputs.length > 0) {
    const isGeneratedVideos = $("dataset").value === "generated_videos";
    const modelCount = state.modelOutputs.length;
    
    modelOutputsHtml += `<div class="model-outputs-title">各模型输出参考 (${modelCount}个模型):</div>`;
    
    state.modelOutputs.forEach((mo, idx) => {
      const enField = stepFields[step] || step;
      const zhField = zhFields[step] || step + "_zh";
      const zhText = mo[zhField] || "";
      const enText = mo[enField] || "";

      // 生成视频显示分数信息
      let scoreHtml = "";
      if (isGeneratedVideos && mo.avg_scores) {
        scoreHtml = `<div class="model-score">分数: ${mo.avg_scores.avg_total_score?.toFixed(1) || "N/A"}</div>`;
      }

      modelOutputsHtml += `<div class="model-output-item clickable" onclick="fillFromModel(${idx}, '${step}', '${zhField}', '${enField}')">
        <div class="model-name">${idx + 1}. ${mo.model || "unknown"} ${scoreHtml}</div>
        <div class="model-output-content">
          ${zhText ? `<div class="model-zh">中文: ${zhText}</div>` : ""}
          ${enText ? `<div class="model-en">英文: ${enText}</div>` : ""}
        </div>
      </div>`;
    });
  } else {
    modelOutputsHtml += `<div class="model-no-output">暂无模型输出数据</div>`;
  }
  modelOutputsHtml += `</div>`;

  const textareaId = step;
  const currentVisibleIndex = visibleSteps.indexOf(step);
  const nextText = currentVisibleIndex < visibleSteps.length - 1 ? "下一步" : "完成";
  
  if ($("step-form")) {
    $("step-form").innerHTML = `
      <div class="step-content">
        <div class="step-label">${STEP_LABELS[step]}</div>
        <textarea id="${textareaId}" placeholder="请输入人工总结..." class="auto-resize">${currentValue}</textarea>
        ${modelOutputsHtml}
      </div>
    `;
  }

  // 自动调整 textarea 高度
  const textarea = $(textareaId);
  if (textarea) {
    autoResizeTextarea(textarea);
    textarea.addEventListener('input', () => autoResizeTextarea(textarea));
  }

  // 更新 header 中的按钮状态
  const prevBtn = $("prev-step");
  const nextBtn = $("next-step");
  if (prevBtn) prevBtn.style.display = currentVisibleIndex > 0 ? "inline-block" : "none";
  if (nextBtn) nextBtn.textContent = nextText;
}

function jumpToStep(targetStep) {
  if (!state.current) return;
  if (targetStep < 0 || targetStep >= STEPS.length || targetStep === state.currentStep) return;
  
  const isRisk = $("risk")?.value === "Yes";
  const visibleSteps = isRisk ? STEPS : ["description"];
  if (!visibleSteps.includes(STEPS[targetStep])) return;
  
  // Save current step value before leaving
  const step = STEPS[state.currentStep];
  const textarea = document.getElementById(step);
  if (textarea) {
    if (!state.current.annotation) state.current.annotation = {};
    state.current.annotation[step] = textarea.value;
  }
  
  state.currentStep = targetStep;
  renderStepForm();
}

function fillFromModel(idx, step, zhField, enField) {
  if (!state.modelOutputs || !state.modelOutputs[idx]) return;
  const mo = state.modelOutputs[idx];
  const text = mo[zhField] || mo[enField] || "";
  if (!text) return;
  
  const textarea = document.getElementById(step);
  if (textarea) {
    textarea.value = text;
    // Also update state
    if (!state.current.annotation) state.current.annotation = {};
    state.current.annotation[step] = text;
  }
}

function changeStep(delta) {
  const isRisk = $("risk")?.value === "Yes";
  const visibleSteps = isRisk ? STEPS : ["description"];
  let newStep = state.currentStep;
  do {
    newStep = newStep + delta;
  } while (newStep >= 0 && newStep < STEPS.length && !visibleSteps.includes(STEPS[newStep]));
  if (newStep < 0 || newStep >= STEPS.length || !visibleSteps.includes(STEPS[newStep])) return;
  
  // Save current step value before leaving
  if (state.current) {
    const step = STEPS[state.currentStep];
    const textarea = document.getElementById(step);
    if (textarea) {
      if (!state.current.annotation) state.current.annotation = {};
      state.current.annotation[step] = textarea.value;
    }
  }
  
  state.currentStep = newStep;
  renderStepForm();
}

async function reportBrokenVideo() {
  if (!state.current) return;
  const videoKey = state.current.video_key;
  if (!confirm(`确认视频 "${videoKey}" 无法加载？`)) return;
  try {
    const res = await fetch("/api/report_broken", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_key: videoKey })
    });
    const data = await res.json();
    if (res.ok) {
      alert("已记录");
    } else {
      alert(data.error || "记录失败");
    }
  } catch (e) {
    alert("记录失败: " + e.message);
  }
}

function currentIndex() {
  return state.videos.findIndex(v => v.video_key === state.current?.video_key);
}

function moveSelection(delta) {
  const index = currentIndex();
  if (index < 0) return;
  const nextIndex = index + delta;
  if (nextIndex < 0 || nextIndex >= state.videos.length) return;
  selectVideo(nextIndex);
}

async function showPrevAnnotation() {
  const el = $("prev-annotation-content");
  el.style.display = el.style.display === "none" ? "block" : "none";
  if (el.textContent) return;
  try {
    const res = await fetch(`/api/my_last_annotation?annotator=${encodeURIComponent(annotator())}`);
    if (!res.ok) {
      el.textContent = "暂无标注记录";
      return;
    }
    const ann = await res.json();
    const parts = [];
    if (ann.risk) parts.push(`风险: ${ann.risk === "Yes" ? "有风险" : "无风险"}`);
    if (ann.risk_subtype) parts.push(`风险类别: ${toChinese("risk_subtype", ann.risk_subtype)}`);
    if (ann.level1_scene) parts.push(`场景: ${toChinese("level1_scene", ann.level1_scene)}`);
    if (ann.level2_subject) parts.push(`主体: ${toChinese("level2_subject", ann.level2_subject)}`);
    if (ann.level3_risk_type) parts.push(`风险类型: ${toChinese("level3_risk_type", ann.level3_risk_type)}`);
    if (ann.risk_localization) parts.push(`风险定位: ${ann.risk_localization}`);
    if (ann.precursor_start_time) parts.push(`风险前兆开始时刻: ${ann.precursor_start_time}`);
    if (ann.description) parts.push(`描述: ${ann.description}`);
    if (ann.solution_for_person) parts.push(`解决方案-对人: ${ann.solution_for_person}`);
    if (ann.solution_for_hazard_source) parts.push(`解决方案-对危险源: ${ann.solution_for_hazard_source}`);
    if (ann.solution_prevent_recurrence) parts.push(`防止复发: ${ann.solution_prevent_recurrence}`);
    if (ann.notes) parts.push(`备注: ${ann.notes}`);
    el.textContent = parts.join(" | ");
  } catch (e) {
    el.textContent = "加载失败";
  }
}

function updateRiskSubtypeVisibility() {
  const label = $("risk-subtype-label");
  if (!label) return;
  const visible = $("risk").value === "Yes";
  label.style.visibility = visible ? "visible" : "hidden";
  label.style.pointerEvents = visible ? "auto" : "none";
}

function updateLevel3CustomVisibility() {
  const customArea = $("level3_custom_area");
  if (!customArea) return;
  const select = $("level3_risk_type");
  customArea.style.display = select?.value === "__custom__" ? "block" : "none";
}

function updateTrimVisibility() {
  const area = $("trim-input-area");
  if (!area) return;
  area.style.display = $("trim_needed").checked ? "block" : "none";
  if (!$("trim_needed").checked) {
    $("trim_segments").value = "";
  }
}

function updateRiskFieldsVisibility() {
  const riskFields = $("risk-fields");
  if (!riskFields) return;
  const isRisk = $("risk")?.value === "Yes";
  riskFields.style.display = isRisk ? "grid" : "none";
}

async function saveCurrent() {
  if (!state.current) return;
  const indexBeforeSave = currentIndex();
  const isGeneratedVideos = $("dataset").value === "generated_videos";

  // 生成视频使用简化的保存逻辑
  if (isGeneratedVideos) {
    const keepDecision = document.querySelector('input[name="keep_decision"]:checked');
    
    const payload = {
      annotator: annotator(),
      risk: keepDecision ? (keepDecision.value === "keep" ? "Yes" : "No") : "",
      notes: "",
      description: "",
      solution_for_person: "",
      solution_for_hazard_source: "",
      solution_prevent_recurrence: "",
      risk_subtype: "",
      level1_scene: "",
      level2_subject: "",
      level3_risk_type: "",
      risk_localization: "",
      precursor_start_time: "",
      trim_segments: "",
    };
    
    const res = await fetch(`/api/videos/${encodeURIComponent(state.current.video_key)}/annotation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      alert(await res.text());
      return;
    }
    state.current.annotation = {
      ...state.current.annotation,
      ...payload,
      annotator: annotator(),
      updated_at: new Date().toISOString()
    };
    renderList();
    const nextIndex = Math.min(indexBeforeSave + 1, state.videos.length - 1);
    if (nextIndex >= 0) {
      selectVideo(nextIndex);
    }
    await Promise.all([loadStats(), loadDeviceStats()]);
    return;
  }

  // Save current step value to state before saving
  const step = STEPS[state.currentStep];
  const textarea = document.getElementById(step);
  if (textarea) {
    if (!state.current.annotation) state.current.annotation = {};
    state.current.annotation[step] = textarea.value;
  }

  const payload = { annotator: annotator() };
  // Read fixed fields from DOM
  ["risk","risk_subtype","level1_scene","level2_subject","level3_risk_type","risk_localization","precursor_start_time","notes"].forEach(k => {
    payload[k] = $(k) ? $(k).value : "";
  });
  if (payload.level3_risk_type === "__custom__") {
    payload.level3_risk_type = $("level3_risk_type_custom")?.value || "";
  }
  // Read step fields from state.current.annotation
  STEPS.forEach(s => {
    payload[s] = (state.current.annotation && state.current.annotation[s]) ? state.current.annotation[s] : "";
  });
  payload.trim_segments = $("trim_needed") && $("trim_needed").checked ? $("trim_segments").value.trim() : "";
  if (payload.risk !== "Yes") {
    payload.risk_subtype = "";
  }
  const res = await fetch(`/api/videos/${encodeURIComponent(state.current.video_key)}/annotation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    alert(await res.text());
    return;
  }
  // Update local state without full reload
  state.current.annotation = {
    ...state.current.annotation,
    ...payload,
    annotator: annotator(),
    updated_at: new Date().toISOString()
  };
  renderList();
  // Move to next video in list
  const nextIndex = await advanceAfterReview(indexBeforeSave);
  if (nextIndex >= 0) {
    selectVideo(nextIndex);
  }
  // Refresh statistics
  await Promise.all([loadStats(), loadDeviceStats()]);
}

async function advanceAfterReview(index) {
  if (index < 0) return -1;
  // Only auto-requeue in the "annotated" view (revisit flow), so already
  // annotated videos get pushed to the back of the loaded list and a new
  // one is pulled in.
  if ($("status").value !== "annotated") {
    return Math.min(index + 1, state.videos.length - 1);
  }
  const reviewed = state.videos[index];
  if (!reviewed) return Math.min(index, state.videos.length - 1);
  const isAbnormal = $("risk-subtype-filter").value === "abnormal";
  const inUnreviewed = state.revisit === "unreviewed";

  if (inUnreviewed && isAbnormal) {
    // In the "unreviewed" tab, the just-tagged video has moved to the
    // "reviewed" bucket on the server, so it should disappear from this
    // list. Remove it and do not append anything.
    state.videos.splice(index, 1);
    // Pull in one replacement so the list stays at its current size.
    const params = new URLSearchParams({
      annotator: annotator(),
      dataset: $("dataset").value,
      video_type: $("video-type").value,
      risk_subtype: $("risk-subtype-filter").value,
      status: $("status").value,
      duration: $("duration-filter").value,
      revisit: state.revisit || "",
      offset: state.videos.length,
      limit: 1,
    });
    try {
      const res = await fetch(`/api/videos?${params}`);
      const data = await res.json();
      const extra = data.videos || [];
      if (extra.length > 0) {
        const existing = new Set(state.videos.map(v => v.video_key));
        for (const v of extra) {
          if (!existing.has(v.video_key)) state.videos.push(v);
        }
      }
    } catch (e) {
      console.error("Failed to fetch next video:", e);
    }
    renderList();
    return Math.min(index, state.videos.length - 1);
  }

  // Default behavior (used by the "全部" and "已标" tabs, and by non-abnormal
  // sub-types): keep the just-reviewed video in the list, push it to the
  // back, and pull in one more so the list size stays the same.
  state.videos.splice(index, 1);
  state.videos.push(reviewed);
  const params = new URLSearchParams({
    annotator: annotator(),
    dataset: $("dataset").value,
    video_type: $("video-type").value,
    risk_subtype: $("risk-subtype-filter").value,
    status: $("status").value,
    duration: $("duration-filter").value,
    revisit: state.revisit || "",
    offset: state.videos.length,
    limit: 1,
  });
  try {
    const res = await fetch(`/api/videos?${params}`);
    const data = await res.json();
    const extra = data.videos || [];
    if (extra.length > 0) {
      const existing = new Set(state.videos.map(v => v.video_key));
      for (const v of extra) {
        if (!existing.has(v.video_key)) state.videos.push(v);
      }
    }
  } catch (e) {
    console.error("Failed to fetch next video:", e);
  }
  renderList();
  return Math.min(index, state.videos.length - 1);
}

async function skipCurrent() {
  if (!state.current) return;
  const index = currentIndex();
  const videoKey = state.current.video_key;

  const res = await fetch(`/api/videos/${encodeURIComponent(videoKey)}/skip`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  if (!res.ok) {
    const err = await res.json();
    alert(err.error || "跳过失败");
    return;
  }

  // In the "annotated" revisit view, push the just-skipped video to the
  // end of the list and pull in a new one (so the list keeps refilling
  // with unseen videos). In other views, drop the skipped video and
  // fetch a replacement to keep the list size stable.
  let nextIndex;
  if ($("status").value === "annotated") {
    nextIndex = await advanceAfterReview(index);
  } else {
    state.videos.splice(index, 1);

    const params = new URLSearchParams({
      annotator: annotator(),
      dataset: $("dataset").value,
      video_type: $("video-type").value,
      risk_subtype: $("risk-subtype-filter").value,
      status: $("status").value,
      duration: $("duration-filter").value,
      revisit: state.revisit || "",
      offset: state.videos.length,
      limit: 1,
    });
    const res2 = await fetch(`/api/videos?${params}`);
    const data = await res2.json();
    const extra = data.videos || [];
    if (extra.length > 0) {
      const existing = new Set(state.videos.map(v => v.video_key));
      for (const v of extra) {
        if (!existing.has(v.video_key)) state.videos.push(v);
      }
    }
    renderList();
    nextIndex = Math.min(index, state.videos.length - 1);
  }

  if (state.videos.length > 0) {
    selectVideo(nextIndex);
  } else {
    state.current = null;
    $("detail").innerHTML = `<div class="empty">当前筛选下没有待标注视频</div>`;
  }
  await Promise.all([loadStats(), loadDeviceStats()]);
}

async function unskipCurrent() {
  if (!state.current) return;
  const videoKey = state.current.video_key;

  const res = await fetch(`/api/videos/${encodeURIComponent(videoKey)}/unskip`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  if (!res.ok) {
    alert("取消失败");
    return;
  }

  await loadVideos();
  const newIndex = state.videos.findIndex(v => v.video_key === videoKey);
  if (newIndex >= 0) {
    selectVideo(newIndex);
  } else if (state.videos.length > 0) {
    selectVideo(0);
  } else {
    state.current = null;
    $("detail").innerHTML = `<div class="empty">当前筛选下没有待标注视频</div>`;
  }
  await Promise.all([loadStats(), loadDeviceStats()]);
}

async function downloadTrimmedVideo() {
  if (!state.current) return;
  const trimSegments = $("trim_segments").value.trim();
  if (!trimSegments) {
    alert("请输入裁切时间段");
    return;
  }

  const btn = $("trim-download");
  btn.disabled = true;
  btn.textContent = "裁切中...";

  try {
    const res = await fetch(`/api/videos/${encodeURIComponent(state.current.video_key)}/trim`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trim_segments: trimSegments }),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(err.error || "裁切失败");
      return;
    }

    const blob = await res.blob();
    const contentDisposition = res.headers.get("Content-Disposition");
    let filename = "trimmed_video.mp4";
    if (contentDisposition) {
      const match = contentDisposition.match(/filename=(.+)/);
      if (match) filename = match[1];
    }

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert("裁切失败: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "裁切并下载";
  }
}

async function exportAnnotations() {
  const res = await fetch(`/api/export?annotator=${encodeURIComponent(annotator())}`);
  const data = await res.json();
  const blob = new Blob([data.content], { type: "application/jsonl" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = data.filename;
  a.click();
  URL.revokeObjectURL(url);
}

async function batchTrim() {
  const btn = $("batch-trim");
  btn.disabled = true;
  btn.textContent = "裁切中...";

  try {
    const dataset = $("dataset").value;
    const res = await fetch(`/api/batch_trim?dataset=${encodeURIComponent(dataset)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const data = await res.json();

    if (!res.ok) {
      showCopyableModal(data.error || "批量裁切失败");
      return;
    }

    let msg = `批量裁切完成！\n\n`;
    msg += `输出目录: ${data.output_dir}\n`;
    msg += `总计: ${data.total} 个视频\n`;
    msg += `成功: ${data.success} 个\n`;
    msg += `失败: ${data.failed} 个\n`;

    if (data.results && data.results.length > 0) {
      const successes = data.results.filter(r => r.status === "success");
      if (successes.length > 0) {
        msg += `\n成功裁切的文件:\n`;
        successes.forEach(r => {
          msg += `  ${r.output}\n`;
        });
      }

      const errors = data.results.filter(r => r.status === "error" || r.status === "skipped");
      if (errors.length > 0) {
        msg += `\n失败/跳过:\n`;
        errors.forEach(r => {
          msg += `  ${r.video_key}: ${r.reason}\n`;
        });
      }
    }

    showCopyableModal(msg);
  } catch (e) {
    showCopyableModal("批量裁切失败: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "批量裁切";
  }
}

function showCopyableModal(text) {
  const modal = document.createElement("div");
  modal.style.cssText = "position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);display:flex;justify-content:center;align-items:center;z-index:1000";
  modal.innerHTML = `
    <div style="background:#fff;padding:24px;border-radius:8px;width:600px;max-height:80vh;display:flex;flex-direction:column">
      <h3 style="margin:0 0 16px">批量裁切结果</h3>
      <textarea readonly style="flex:1;min-height:300px;padding:12px;border:1px solid #ddd;border-radius:4px;font-family:monospace;font-size:13px;resize:none">${text}</textarea>
      <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:16px">
        <button onclick="navigator.clipboard.writeText(this.closest('div[style*=fixed]').querySelector('textarea').value);this.textContent='已复制!'" style="padding:8px 16px;background:#3b82f6;color:#fff;border:none;border-radius:6px;cursor:pointer">复制全部</button>
        <button onclick="this.closest('div[style*=fixed]').remove()" style="padding:8px 16px;background:#6b7280;color:#fff;border:none;border-radius:6px;cursor:pointer">关闭</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
}

$("reload").onclick = async () => {
  if ($("dataset").value === "generated_videos") {
    await syncGeneratedVideos();
  }
  loadVideos();
};
$("export").onclick = exportAnnotations;
$("batch-trim").onclick = batchTrim;
$("stats-btn").onclick = () => window.open('/statistics', '_blank');
$("dataset").onchange = async () => {
  // 显示/隐藏排序选项
  const sortByEl = $("sort-by");
  if ($("dataset").value === "generated_videos") {
    sortByEl.style.display = "inline-block";
    // 自动同步生成视频结果
    await syncGeneratedVideos();
  } else {
    sortByEl.style.display = "none";
    sortByEl.value = "";
  }
  loadVideos();
  resizeDatasetSelect();
};
$("sort-by").onchange = () => loadVideos();
$("video-type").onchange = () => {
  filterAndRenderDatasets();
  if ($("video-type").value) {
    $("status").value = "annotated";
  }
  $("risk-subtype-filter").style.display = $("video-type").value === "real" ? "inline-block" : "none";
  if ($("video-type").value !== "real") $("risk-subtype-filter").value = "";
  loadVideos();
};
$("risk-subtype-filter").onchange = () => { state.revisit = "unreviewed"; loadVideos(); };
$("status").onchange = () => { state.revisit = "unreviewed"; loadVideos(); };
$("duration-filter").onchange = () => loadVideos();

// 搜索框：输入时延迟搜索，回车立即搜索
$("search").oninput = () => {
  const term = $("search").value.trim();
  state.searchTerm = term;
  if (term && $("status").value === "pending") {
    $("status").value = "";
    state.revisit = "unreviewed";
  }
  clearTimeout(state.searchTimer);
  state.searchTimer = setTimeout(() => {
    loadVideos();
  }, 300);
};
$("search").onkeydown = (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    clearTimeout(state.searchTimer);
    state.searchTerm = $("search").value.trim();
    if (state.searchTerm && $("status").value === "pending") {
      $("status").value = "";
      state.revisit = "unreviewed";
    }
    loadVideos();
  }
};

// 同步生成视频结果
async function syncGeneratedVideos() {
  try {
    const res = await fetch("/api/sync_generated_videos", { method: "POST" });
    const data = await res.json();
    if (data.success && data.stats) {
      console.log(`Synced generated videos: ${data.stats.imported} imported`);
    }
  } catch (e) {
    console.error("Failed to sync generated videos:", e);
  }
}

// Load current user info
async function loadUserInfo() {
  const res = await fetch("/api/me");
  const data = await res.json();
  if (data.logged_in) {
    state.currentUsername = data.username;
    $("current-user").textContent = data.username;
    $("manage-users").style.display = data.username === "admin" ? "inline-block" : "none";
    $("git-backup").style.display = ["ligaoxiang", "caiqingyuan"].includes(data.username) ? "inline-block" : "none";
  }
}

// Logout
$("logout").onclick = async () => {
  await fetch("/api/logout", { method: "POST" });
  window.location.href = "/login";
};

$("git-backup").onclick = async () => {
  const btn = $("git-backup");
  btn.disabled = true;
  btn.textContent = "备份中...";
  try {
    const res = await fetch("/api/git_backup", { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || "备份失败");
    } else {
      alert("备份成功: " + data.message);
    }
  } catch (e) {
    alert("备份失败: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Git备份";
  }
};

// User management modal
$("manage-users").onclick = async () => {
  const res = await fetch("/api/users");
  const users = await res.json();
  const modal = document.createElement("div");
  modal.style.cssText = "position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);display:flex;justify-content:center;align-items:center;z-index:1000";
  modal.innerHTML = `
    <div style="background:#fff;padding:24px;border-radius:8px;width:400px;max-height:80vh;overflow:auto">
      <h3 style="margin:0 0 16px">用户管理</h3>
      <div id="user-list" style="margin-bottom:16px">
        ${users.map(u => `
          <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #eee">
            <span>${u.username}</span>
            ${u.username !== 'admin' ? `<button onclick="deleteUser(${u.id}, this)" style="font-size:12px;padding:2px 8px;background:#dc2626">删除</button>` : ''}
          </div>
        `).join('')}
      </div>
      <h4 style="margin:0 0 8px">添加用户</h4>
      <input id="new-username" placeholder="用户名" style="width:100%;padding:8px;margin-bottom:8px;border:1px solid #ccc;border-radius:4px" />
      <input id="new-password" type="password" placeholder="密码(至少6位)" style="width:100%;padding:8px;margin-bottom:8px;border:1px solid #ccc;border-radius:4px" />
      <div style="display:flex;gap:8px;justify-content:flex-end">
        <button onclick="this.closest('div[style*=fixed]').remove()" class="secondary">关闭</button>
        <button onclick="addUser()">添加</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
};

async function addUser() {
  const username = document.getElementById("new-username").value.trim();
  const password = document.getElementById("new-password").value;
  if (!username || !password) { alert("请输入用户名和密码"); return; }
  if (password.length < 6) { alert("密码至少6位"); return; }
  const res = await fetch("/api/users", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password })
  });
  const data = await res.json();
  if (res.ok) {
    document.querySelector("div[style*=fixed]").remove();
    $("manage-users").click();
  } else {
    alert(data.error || "添加失败");
  }
}

async function deleteUser(userId, btn) {
  if (!confirm("确定删除此用户？")) return;
  await fetch(`/api/users/${userId}`, { method: "DELETE" });
  btn.parentElement.remove();
}

Promise.all([loadTaxonomy(), loadDatasets()]).then(loadVideos).then(loadDeviceStats).then(loadUserInfo);

let evtSource = new EventSource("/api/stream");
evtSource.onmessage = (e) => {
  if (e.data === "refresh") {
    loadStats();
    loadDeviceStats();
  }
};
evtSource.onerror = () => {
  // Silently ignore SSE errors - it's not critical
};

window.addEventListener("beforeunload", () => {
  navigator.sendBeacon("/api/current_viewing", new Blob([JSON.stringify({})], { type: "application/json" }));
});

async function loadOnlineUsers() {
  try {
    const res = await fetch("/api/online_users");
    const data = await res.json();
    if (data.users && data.users.length > 0) {
      $("online-users").textContent = "在线用户: " + data.users.join(", ");
    } else {
      $("online-users").textContent = "";
    }
  } catch (e) {}
}
setInterval(loadOnlineUsers, 5000);
loadOnlineUsers();
