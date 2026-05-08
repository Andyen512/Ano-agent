const state = { videos: [], current: null, taxonomy: {}, stats: null, deviceStats: null, currentUsername: null, latestRequestId: 0, currentStep: 0, modelOutputs: null, chineseOutputs: null };

const $ = (id) => document.getElementById(id);

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
  const datasets = await res.json();
  $("dataset").innerHTML = `<option value="">全部数据集</option>` + datasets.map(d => `<option value="${d.dataset}">${d.dataset} (${d.count})</option>`).join("");
}

async function loadTaxonomy() {
  const res = await fetch("/api/taxonomy");
  state.taxonomy = await res.json();
}

async function loadStats() {
  const res = await fetch(`/api/stats?annotator=${encodeURIComponent(annotator())}`);
  const s = await res.json();
  state.stats = s;
  $("stats").textContent = `总数 ${s.total} | 已标注 ${s.annotated || 0} | 待标 ${s.pending || s.total} | 我已标 ${s.user_annotated || 0} | 模型风险 ${s.risk?.Yes || 0} | 模型无风险 ${s.risk?.No || 0}`;
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
  const params = new URLSearchParams({
    annotator: annotator(),
    dataset: $("dataset").value,
    status: $("status").value,
    offset: append ? state.videos.length : 0,
    limit: "50",
  });
  const res = await fetch(`/api/videos?${params}`);
  if (requestId !== state.latestRequestId) return;
  const newVideos = await res.json();
  if (requestId !== state.latestRequestId) return;
  if (append) {
    state.videos.push(...newVideos);
  } else {
    state.videos = newVideos;
  }
  renderList(append);
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

function renderList(append = false) {
  const html = state.videos.map((v, i) => {
    const riskClass = v.model_risk === "Yes" ? "risk-yes" : "risk-no";
    const hasTrim = v.annotation?.trim_segments;
    const annotator = v.annotation?.annotator;
    return `<div class="item ${state.current?.video_key === v.video_key ? "active" : ""}" data-i="${i}">
      <strong>${v.video_key}</strong>
      <span class="badge ${riskClass}">模型: ${v.model_risk}</span>
      <span class="badge">${v.votes_cast || 0} votes | Yes ${v.yes_count || 0} / No ${v.no_count || 0}</span>
      ${v.annotation ? `<span class="badge">已标${annotator ? '(' + annotator + ')' : ''}</span>` : ""}
      ${hasTrim ? `<span class="badge trim-badge">需裁切</span>` : ""}
    </div>`;
  }).join("");
  
  if (append) {
    $("list").insertAdjacentHTML("beforeend", html);
  } else {
    $("list").innerHTML = html + `<button id="load-more" class="secondary" style="width:100%;margin-top:8px">加载更多</button>`;
    $("load-more").onclick = () => loadVideos(true);
  }
  [...document.querySelectorAll(".item")].forEach(el => {
    el.onclick = (e) => {
      // 如果用户是在选择文字，不触发选择
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
  level3_risk_type: { "fall/instability": "跌倒/不稳", "heat/fire source": "热源/火灾", "collision/crush injury": "碰撞/挤压伤", "sharp-object danger": "锐器危险", "electrical safety": "电气安全", "poisoning/accidental ingestion": "中毒/误食", "interpersonal conflict": "人际冲突", "animal attack/biosecurity risk": "动物攻击/生物安全风险", "None": "无" },
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

function progressBar(label, progress) {
  const total = progress?.total || 0;
  const annotated = progress?.annotated || 0;
  const pct = total > 0 ? Math.round((annotated / total) * 100) : 0;
  return `<div class="progress-row">
    <div class="progress-label"><span>${label}</span><span>${annotated}/${total} (${pct}%)</span></div>
    <div class="progress-track"><div class="progress-fill" style="width: ${pct}%"></div></div>
  </div>`;
}

function progressPanel() {
  const progress = state.stats?.progress || {};
  const devices = state.deviceStats?.devices || [];
  return `<div class="progress-panel">
    ${progressBar("风险视频标注进度", progress.risk)}
    ${progressBar("无风险视频标注进度", progress.normal)}
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

function selectVideo(index) {
  const v = state.videos[index];
  state.current = v;
  state.currentStep = 0;
  state.modelOutputs = null;
  state.chineseOutputs = null;
  const ann = v.annotation || {};
  const value = (key) => ann[key] ?? modelValue(v, key);
  
  $("detail").innerHTML = `
    <div class="grid">
      <div>
        <video controls autoplay loop muted preload="auto" src="${v.media_url}" onerror="this.outerHTML='<div class=\'video-error\'>视频加载失败</div>'"></video>
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
          </div>
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
            <label id="risk-subtype-label">风险类别
              <select id="risk_subtype">
                <option value="abnormal" ${value("risk_subtype") === "abnormal" ? "selected" : ""}>${toChinese("risk_subtype","abnormal")}</option>
                <option value="risk_only" ${value("risk_subtype") === "risk_only" ? "selected" : ""}>${toChinese("risk_subtype","risk_only")}</option>
              </select>
            </label>
            ${selectField("level1_scene", "Level 1 场景", value("level1_scene"), state.taxonomy.level1_scene || [])}
            ${selectField("level2_subject", "Level 2 主体", value("level2_subject"), state.taxonomy.level2_subject || [])}
            ${selectField("level3_risk_type", "Level 3 风险类型", value("level3_risk_type"), state.taxonomy.level3_risk_type || [])}
          </div>
          <label class="inline-label"><span class="label-title">风险定位 <small>(如: 5,14)</small></span><input id="risk_localization" value="${value("risk_localization") || ""}" /></label>
          <label class="inline-label">备注<textarea id="notes" class="small-textarea">${ann.notes || ""}</textarea></label>
        </div>
      </div>
    </div>
    ${renderAllAnnotations(v.annotations || [])}`;
  
  $("save").onclick = saveCurrent;
  $("prev-video").onclick = () => moveSelection(-1);
  $("next-video").onclick = () => moveSelection(1);
  $("prev-step").onclick = () => changeStep(-1);
  $("next-step").onclick = () => changeStep(1);
  $("trim_needed").onchange = updateTrimVisibility;
  
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
  const step = STEPS[state.currentStep];
  const ann = state.current.annotation || {};
  const currentValue = ann[step] || "";

  let indicatorHtml = `<div class="step-dots">`;
  STEPS.forEach((s, i) => {
    indicatorHtml += `<span class="step-dot ${i === state.currentStep ? 'active' : ''} ${ann[s] ? 'completed' : ''}" data-step="${i}" onclick="jumpToStep(${i})">${STEP_LABELS[s].slice(0, 4)}</span>`;
  });
  indicatorHtml += `</div>`;
  $("step-indicator").innerHTML = indicatorHtml;

  let modelOutputsHtml = `<div class="model-outputs">`;
  modelOutputsHtml += `<div class="model-outputs-title">各模型输出参考 (5个模型):</div>`;

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
    state.modelOutputs.forEach((mo, idx) => {
      const enField = stepFields[step] || step;
      const zhField = zhFields[step] || step + "_zh";
      const zhText = mo[zhField] || "";
      const enText = mo[enField] || "";

      modelOutputsHtml += `<div class="model-output-item clickable" onclick="fillFromModel(${idx}, '${step}', '${zhField}', '${enField}')">
        <div class="model-name">${idx + 1}. ${mo.model || "unknown"}</div>
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
  const prevDisplay = state.currentStep > 0 ? "inline-block" : "none";
  const nextText = state.currentStep < STEPS.length - 1 ? "下一步" : "完成";
  
  $("step-form").innerHTML = `
    <div class="step-content">
      <div class="step-label">${STEP_LABELS[step]}</div>
      <textarea id="${textareaId}" placeholder="请输入人工总结...">${currentValue}</textarea>
      ${modelOutputsHtml}
    </div>
  `;

  // 更新 header 中的按钮状态
  const prevBtn = $("prev-step");
  const nextBtn = $("next-step");
  if (prevBtn) prevBtn.style.display = state.currentStep > 0 ? "inline-block" : "none";
  if (nextBtn) nextBtn.textContent = state.currentStep < STEPS.length - 1 ? "下一步" : "完成";
}

function jumpToStep(targetStep) {
  if (!state.current) return;
  if (targetStep < 0 || targetStep >= STEPS.length || targetStep === state.currentStep) return;
  
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
  const newStep = state.currentStep + delta;
  if (newStep < 0 || newStep >= STEPS.length) return;
  
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

function updateTrimVisibility() {
  const area = $("trim-input-area");
  if (!area) return;
  area.style.display = $("trim_needed").checked ? "block" : "none";
  if (!$("trim_needed").checked) {
    $("trim_segments").value = "";
  }
}

async function saveCurrent() {
  if (!state.current) return;
  const indexBeforeSave = currentIndex();

  // Save current step value to state before saving
  const step = STEPS[state.currentStep];
  const textarea = document.getElementById(step);
  if (textarea) {
    if (!state.current.annotation) state.current.annotation = {};
    state.current.annotation[step] = textarea.value;
  }

  const payload = { annotator: annotator() };
  // Read fixed fields from DOM
  ["risk","risk_subtype","level1_scene","level2_subject","level3_risk_type","risk_localization","notes"].forEach(k => {
    payload[k] = $(k) ? $(k).value : "";
  });
  // Read step fields from state.current.annotation
  STEPS.forEach(s => {
    payload[s] = (state.current.annotation && state.current.annotation[s]) ? state.current.annotation[s] : "";
  });
  payload.trim_segments = $("trim_needed") && $("trim_needed").checked ? $("trim_segments").value.trim() : "";
  if (payload.risk !== "Yes") {
    payload.risk_subtype = "";
  }
  const res = await fetch(`/api/videos/${state.current.video_key}/annotation`, {
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
  const nextIndex = Math.min(indexBeforeSave + 1, state.videos.length - 1);
  if (nextIndex >= 0) {
    selectVideo(nextIndex);
  }
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
    const res = await fetch(`/api/videos/${state.current.video_key}/trim`, {
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
    const res = await fetch(`/api/batch_trim?annotator=${encodeURIComponent(annotator())}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const data = await res.json();

    if (!res.ok) {
      alert(data.error || "批量裁切失败");
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
        successes.slice(0, 10).forEach(r => {
          msg += `  ${r.output}\n`;
        });
        if (successes.length > 10) {
          msg += `  ... 等 ${successes.length} 个文件\n`;
        }
      }

      const errors = data.results.filter(r => r.status === "error" || r.status === "skipped");
      if (errors.length > 0) {
        msg += `\n失败/跳过:\n`;
        errors.slice(0, 5).forEach(r => {
          msg += `  ${r.video_key}: ${r.reason}\n`;
        });
        if (errors.length > 5) {
          msg += `  ... 等 ${errors.length} 个\n`;
        }
      }
    }

    alert(msg);
  } catch (e) {
    alert("批量裁切失败: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "批量裁切";
  }
}

$("reload").onclick = () => loadVideos();
$("export").onclick = exportAnnotations;
$("batch-trim").onclick = batchTrim;
$("dataset").onchange = () => loadVideos();
$("status").onchange = () => loadVideos();

// Load current user info
async function loadUserInfo() {
  const res = await fetch("/api/me");
  const data = await res.json();
  if (data.logged_in) {
    state.currentUsername = data.username;
    $("current-user").textContent = data.username;
    $("manage-users").style.display = data.username === "admin" ? "inline-block" : "none";
  }
}

// Logout
$("logout").onclick = async () => {
  await fetch("/api/logout", { method: "POST" });
  window.location.href = "/login";
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
    loadVideos();
  }
};
evtSource.onerror = () => {
  // Silently ignore SSE errors - it's not critical
};

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
