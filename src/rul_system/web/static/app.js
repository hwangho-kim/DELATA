const state = {
  file: null,
  data: null,
  activeSensor: null,
  visibility: { raw: true, smooth: true, fit: true, future: true },
  loadingTimer: null,
  isAnalyzing: false,
  isLoadingSample: false,
};

const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 3) => value === null || value === undefined || Number.isNaN(Number(value)) ? "-" : Number(value).toLocaleString("ko-KR", { maximumFractionDigits: digits });
const fmtDate = (value, withTime = false) => {
  if (!value) return "산출 불가";
  const date = new Date(value);
  return withTime ? date.toLocaleString("ko-KR", { hour12: false }) : date.toLocaleDateString("ko-KR");
};
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));

function setFile(file) {
  state.file = file;
  $("fileLabel").textContent = file ? file.name : "FDC 파일 선택";
  $("analyzeButton").disabled = !file;
  $("errorBanner").hidden = true;
}

async function loadSample(mode) {
  if (state.isAnalyzing || state.isLoadingSample) return;
  state.isLoadingSample = true;
  setActionButtons(true);
  try {
    const response = await fetch(`/api/sample?mode=${encodeURIComponent(mode)}`);
    if (!response.ok) throw new Error("샘플 데이터를 불러오지 못했습니다.");
    const blob = await response.blob();
    setFile(new File([blob], `synthetic_${mode}.csv`, { type: "text/csv" }));
    state.isLoadingSample = false;
    await analyze();
  } catch (error) {
    state.isLoadingSample = false;
    setActionButtons(false);
    showError(error.message);
  }
}

function setActionButtons(busy) {
  $("analyzeButton").disabled = busy || !state.file;
  document.querySelectorAll("[data-sample]").forEach(button => { button.disabled = busy; });
  $("fileInput").disabled = busy;
}

function querySettings() {
  const models = [...document.querySelectorAll("#modelOptions input:checked")].map(input => input.value);
  if (!models.length) throw new Error("AutoML 후보 모델을 하나 이상 선택하세요.");
  return new URLSearchParams({
    filename: state.file.name,
    timestamp_column: $("timestampColumn").value.trim(),
    sensor_columns: $("sensorColumns").value.trim(),
    ewma_span: $("ewmaSpan").value,
    rolling_window: $("rollingWindow").value,
    baseline_fraction: $("baselineFraction").value,
    warning_sigma: $("warningSigma").value,
    failure_sigma: $("failureSigma").value,
    consecutive_points: $("consecutivePoints").value,
    max_extrapolation_days: $("maxDays").value,
    enabled_models: models.join(","),
  });
}

const loadingStages = [
  ["입력 데이터 검증 중", "시간축, 결측값, 센서 열을 확인하고 있습니다."],
  ["신호 노이즈 저감 중", "Median 필터와 EWMA로 열화 추세를 복원하고 있습니다."],
  ["열화 시작점 탐지 중", "2σ 연속 신호와 변곡점을 탐색하고 있습니다."],
  ["AutoML 모델 비교 중", "시계열 교차검증으로 궤적 모델을 평가하고 있습니다."],
  ["고장 교차점 계산 중", "3σ 관리 한계와 외삽 곡선의 최초 교차점을 계산합니다."],
];

function startLoading() {
  $("emptyState").hidden = true; $("results").hidden = true; $("loadingState").hidden = false;
  let index = 0;
  const update = () => {
    $("loadingTitle").textContent = loadingStages[index][0];
    $("loadingDetail").textContent = loadingStages[index][1];
    $("loadingProgress").style.width = `${15 + index * 19}%`;
    index = Math.min(index + 1, loadingStages.length - 1);
  };
  update(); state.loadingTimer = setInterval(update, 900);
}
function stopLoading() { clearInterval(state.loadingTimer); state.loadingTimer = null; $("loadingState").hidden = true; }
function showError(message) { stopLoading(); $("errorBanner").textContent = message; $("errorBanner").hidden = false; $("emptyState").hidden = false; }

async function analyze() {
  if (!state.file || state.isAnalyzing) return;
  state.isAnalyzing = true;
  try {
    const params = querySettings(); startLoading(); setActionButtons(true); $("errorBanner").hidden = true;
    const response = await fetch(`/api/analyze?${params}`, { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: state.file });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "분석을 완료하지 못했습니다.");
    state.data = payload; state.activeSensor = payload.화면.주요_위험센서 || Object.keys(payload.센서)[0];
    stopLoading(); renderAll(); $("results").hidden = false; $("downloadButton").disabled = false;
  } catch (error) { showError(error.message); }
  finally { state.isAnalyzing = false; setActionButtons(false); }
}

function renderAll() {
  const view = state.data.화면; const summary = state.data.요약;
  $("systemStatus").textContent = view.상태;
  const banner = $("statusBanner"); banner.className = `status-banner ${view.상태_톤}`;
  $("statusNarrative").textContent = view.예상_고장일 ? `${view.주요_위험센서} 센서가 가장 먼저 고장 한계에 도달할 것으로 예측됩니다.` : "현재 데이터에서는 신뢰 가능한 고장 교차 시점이 확인되지 않았습니다.";
  $("failureDate").textContent = view.예상_고장일 || "산출 불가"; $("failureTime").textContent = fmtDate(view.예상_고장시각, true);
  $("rulDays").textContent = view.RUL_일 === null ? "산출 불가" : `${fmt(view.RUL_일)}일`; $("rulHours").textContent = view.RUL_시간 === null ? "관측 가능한 열화 없음" : `${fmt(view.RUL_시간, 1)}시간`;
  $("criticalSensor").textContent = view.주요_위험센서 || "없음"; $("activeSensorCount").textContent = `열화 센서 ${summary.탐지된_열화센서.length}개`;
  $("analysisDate").textContent = fmtDate(view.분석_기준시각); $("analysisDuration").textContent = `${fmt(view.분석_소요초, 2)}초 소요`;
  $("analysisStamp").textContent = `기준 ${fmtDate(view.분석_기준시각, true)}`;
  renderProcess(); renderSensorSelectors(); renderOverview(); renderQuality(); renderWarnings(); renderSensorDetail(); renderAutoML(); renderRaw();
  requestAnimationFrame(() => { drawHealthChart(); drawSensorChart($("overviewChart"), state.activeSensor); });
}

function renderProcess() {
  $("processSteps").innerHTML = state.data.분석_단계.map(step => `<article class="process-step"><div class="step-top"><span class="step-number">${step.단계}</span><span class="step-state ${step.상태.replaceAll(" ", "-")}">${escapeHtml(step.상태)}</span></div><strong>${escapeHtml(step.이름)}</strong><p>${escapeHtml(step.설명)}</p></article>`).join("");
}

function renderSensorSelectors() {
  const sensors = Object.keys(state.data.센서);
  [$("overviewSensor"), $("automlSensor")].forEach(select => {
    select.innerHTML = sensors.map(sensor => `<option value="${escapeHtml(sensor)}" ${sensor === state.activeSensor ? "selected" : ""}>${escapeHtml(sensor)}</option>`).join("");
  });
  $("sensorButtons").innerHTML = sensors.map(sensor => `<button class="${sensor === state.activeSensor ? "active" : ""}" data-sensor="${escapeHtml(sensor)}">${escapeHtml(sensor)}</button>`).join("");
}

const legendDefs = [
  ["raw","원시 FDC","#aeb9c7",false],["smooth","EWMA 추세","#2767c8",false],["fit","적합 곡선","#cc8b16",false],["future","외삽 예측","#b24c79",true],["warning","2σ 경고","#cc8b16",true],["failure","3σ 고장","#c43d4a",true],["onset","열화 시작점","#7751b6",true],
];
function legendHtml(interactive=false) { return legendDefs.map(([key,label,color,dashed]) => `<button ${interactive && ["raw","smooth","fit","future"].includes(key) ? `data-series="${key}"` : "disabled"} class="legend-item ${dashed ? "dashed" : ""} ${!interactive || state.visibility[key] !== false ? "on" : ""}" style="--legend-color:${color}"><i></i>${label}</button>`).join(""); }

function renderOverview() {
  $("overviewLegend").innerHTML = legendHtml(false); const sensor = state.data.센서[state.activeSensor];
  $("overviewFormula").textContent = sensor.모델식 ? `모델식: ${sensor.모델식}` : `모델식 없음 — ${sensor.상태}`;
  $("healthDescription").textContent = state.data.건강_이상점수.설명;
}

function renderQuality() {
  const quality = state.data.데이터_품질; $("sourceFileName").textContent = quality.파일명;
  const totalMissing = Object.values(quality.센서별_결측수).reduce((a,b)=>a+b,0);
  const items = [["입력 행",fmt(quality.입력_행수,0)],["분석 센서",`${quality.센서_수}개`],["수집 주기",quality.중앙_수집주기_시간 === null ? "-" : `${fmt(quality.중앙_수집주기_시간,2)}시간`],["결측 계측값",fmt(totalMissing,0)],["중복 시각",fmt(quality.중복_시각수,0)],["무효 시각",fmt(quality.무효_시각수,0)],["데이터 시작",fmtDate(quality.시작시각)],["데이터 종료",fmtDate(quality.종료시각)]];
  $("qualityGrid").innerHTML = items.map(([label,value])=>`<div class="quality-item"><span>${label}</span><strong>${value}</strong></div>`).join("");
}
function renderWarnings() { const warnings = state.data.주의사항.length ? state.data.주의사항 : ["통계 모델 결과는 설비별 물리 고장 기준과 PM 이력으로 최종 검토해야 합니다.","모든 궤적과 관리 한계는 원시 계측 단위로 표시됩니다."]; $("warningList").innerHTML = warnings.map(item=>`<li>${escapeHtml(item)}</li>`).join(""); }

function renderSensorDetail() {
  const sensor = state.data.센서[state.activeSensor]; $("sensorTitle").textContent = `${state.activeSensor} 열화 상세`; $("sensorStatus").textContent = sensor.상태; $("sensorFormula").textContent = sensor.모델식 ? `모델식: ${sensor.모델식}` : `모델식 없음 — ${sensor.상태}`;
  $("sensorLegend").innerHTML = legendHtml(true);
  const details = [["열화 방향",sensor.열화_방향],["건강 기준 평균",fmt(sensor.기준_평균,6)],["기준 표준편차",fmt(sensor.기준_표준편차,6)],["2σ 경고 한계",fmt(sensor.경고_한계,6)],["3σ 고장 한계",fmt(sensor.고장_한계,6)],["열화 시작",fmtDate(sensor.열화_시작시각,true)],["2σ 최초 확인",fmtDate(sensor["2시그마_최초확인시각"],true)],["고장 교차",fmtDate(sensor.예상_고장시각,true)]];
  $("thresholdDetails").innerHTML = details.map(([k,v])=>`<dt>${k}</dt><dd>${v}</dd>`).join("");
  $("featureGrid").innerHTML = Object.entries(sensor.최신_특징).map(([k,v])=>`<div class="feature-item"><span>${k.replaceAll("_"," ")}</span><strong>${fmt(v,6)}</strong></div>`).join("");
  const metrics = Object.entries(sensor.평가지표 || {}); $("metricGrid").innerHTML = metrics.length ? metrics.map(([k,v])=>`<div class="feature-item"><span>${k.replaceAll("_"," ")}</span><strong>${fmt(v,5)}</strong></div>`).join("") : `<div class="feature-item"><span>모델</span><strong>해당 없음</strong></div>`;
  requestAnimationFrame(()=>drawSensorChart($("sensorChart"), state.activeSensor, true));
}

function renderAutoML() {
  const sensorName = $("automlSensor").value || state.activeSensor; const sensor = state.data.센서[sensorName]; const rows = sensor.AutoML_순위표 || [];
  $("modelSummary").innerHTML = sensor.모델 ? `<strong>선택 모델 · ${escapeHtml(sensor.모델)}</strong><span>${escapeHtml(sensor.모델식 || "")}</span>` : `<strong>모델링 대상 없음</strong><span>${escapeHtml(sensor.상태)}</span>`;
  $("leaderboardBody").innerHTML = rows.length ? rows.map((row,index)=>`<tr><td>${index+1}</td><td>${escapeHtml(row.모델)}</td><td title="${escapeHtml(JSON.stringify(row.파라미터))}">${escapeHtml(JSON.stringify(row.파라미터))}</td><td>${fmt(row.교차검증_RMSE,6)}</td><td>${fmt(row.교차검증_R2,5)}</td><td>${fmt(row.AIC,3)}</td><td>${fmt(row.BIC,3)}</td><td>${fmt(row.선택_점수,5)}</td></tr>`).join("") : `<tr><td colspan="8">이 센서는 AutoML 모델링 대상이 아닙니다.</td></tr>`;
}

function renderRaw() {
  const preview = state.data.원시_미리보기; $("rawHead").innerHTML = `<tr>${preview.열.map(col=>`<th>${escapeHtml(col)}</th>`).join("")}</tr>`;
  $("rawBody").innerHTML = preview.행.map(row=>`<tr>${preview.열.map(col=>`<td>${col === preview.열[0] ? escapeHtml(row[col]) : fmt(row[col],6)}</td>`).join("")}</tr>`).join("");
}

function setupCanvas(canvas) { const rect = canvas.getBoundingClientRect(); const ratio = window.devicePixelRatio || 1; canvas.width = Math.max(300, rect.width * ratio); canvas.height = Math.max(220, rect.height * ratio); const ctx = canvas.getContext("2d"); ctx.setTransform(ratio,0,0,ratio,0,0); return {ctx,w:rect.width,h:rect.height}; }
function linePath(ctx, points, color, width=1.5, dash=[]) { ctx.beginPath(); ctx.strokeStyle=color; ctx.lineWidth=width; ctx.setLineDash(dash); let started=false; points.forEach(([x,y])=>{ if(y===null||!Number.isFinite(y)){started=false;return;} if(!started){ctx.moveTo(x,y);started=true;}else ctx.lineTo(x,y); }); ctx.stroke(); ctx.setLineDash([]); }

function drawSensorChart(canvas, sensorName) {
  if (!state.data || !sensorName || canvas.offsetParent === null) return;
  const sensor = state.data.센서[sensorName]; const {ctx,w,h}=setupCanvas(canvas); const margin={l:54,r:18,t:18,b:34}; const pw=w-margin.l-margin.r, ph=h-margin.t-margin.b;
  const observed = sensor.관측.시간.map((t,i)=>({t:+new Date(t),raw:sensor.관측.원시값[i],smooth:sensor.관측.평활값[i]})); const lastObserved=observed.at(-1).t; const firstObserved=observed[0].t; const obsSpan=lastObserved-firstObserved;
  let future=sensor.외삽.시간.map((t,i)=>({t:+new Date(t),v:sensor.외삽.값[i]})); if(!sensor.예상_고장시각) future=future.filter(p=>p.t<=lastObserved+Math.max(obsSpan,30*86400000));
  const fit=sensor.적합.시간.map((t,i)=>({t:+new Date(t),v:sensor.적합.값[i]})); const allTimes=[...observed.map(p=>p.t),...future.map(p=>p.t)]; const xMin=Math.min(...allTimes),xMax=Math.max(...allTimes);
  const values=[...observed.flatMap(p=>[p.raw,p.smooth]),...fit.map(p=>p.v),...future.map(p=>p.v),sensor.경고_한계,sensor.고장_한계].filter(Number.isFinite); let yMin=Math.min(...values),yMax=Math.max(...values); const pad=Math.max((yMax-yMin)*.1,Math.abs(yMax)*.0005,1e-6); yMin-=pad;yMax+=pad;
  const sx=t=>margin.l+(t-xMin)/(xMax-xMin||1)*pw, sy=v=>margin.t+(yMax-v)/(yMax-yMin||1)*ph;
  ctx.clearRect(0,0,w,h); ctx.fillStyle="#fff";ctx.fillRect(0,0,w,h); ctx.font="10px ui-monospace, monospace";ctx.fillStyle="#718095";ctx.strokeStyle="#e4e9ef";ctx.lineWidth=1;
  for(let i=0;i<5;i++){const y=margin.t+ph*i/4;ctx.beginPath();ctx.moveTo(margin.l,y);ctx.lineTo(w-margin.r,y);ctx.stroke();const value=yMax-(yMax-yMin)*i/4;ctx.fillText(fmt(value,4),3,y+3);} for(let i=0;i<5;i++){const x=margin.l+pw*i/4;ctx.beginPath();ctx.moveTo(x,margin.t);ctx.lineTo(x,h-margin.b);ctx.stroke();const d=new Date(xMin+(xMax-xMin)*i/4);ctx.fillText(`${d.getMonth()+1}/${d.getDate()}`,x-12,h-10);}
  if(future.length){ctx.fillStyle="rgba(178,76,121,.045)";ctx.fillRect(sx(lastObserved),margin.t,w-margin.r-sx(lastObserved),ph);}
  [[sensor.경고_한계,"#cc8b16"],[sensor.고장_한계,"#c43d4a"]].forEach(([value,color])=>linePath(ctx,[[margin.l,sy(value)],[w-margin.r,sy(value)]],color,1.2,[6,4]));
  if(sensor.열화_시작시각){const x=sx(+new Date(sensor.열화_시작시각));linePath(ctx,[[x,margin.t],[x,h-margin.b]],"#7751b6",1.2,[3,4]);}
  if(state.visibility.raw) linePath(ctx,observed.map(p=>[sx(p.t),p.raw===null?null:sy(p.raw)]),"#aeb9c7",1);
  if(state.visibility.smooth) linePath(ctx,observed.map(p=>[sx(p.t),p.smooth===null?null:sy(p.smooth)]),"#2767c8",2.2);
  if(state.visibility.fit) linePath(ctx,fit.map(p=>[sx(p.t),sy(p.v)]),"#cc8b16",2);
  if(state.visibility.future) linePath(ctx,future.map(p=>[sx(p.t),sy(p.v)]),"#b24c79",2,[7,4]);
  ctx.strokeStyle="#aeb8c4";ctx.strokeRect(margin.l,margin.t,pw,ph);
}

function drawHealthChart() {
  const canvas=$("healthChart"); if(!state.data||canvas.offsetParent===null)return; const {ctx,w,h}=setupCanvas(canvas); const payload=state.data.건강_이상점수; const pts=payload.시간.map((t,i)=>({t:+new Date(t),v:payload.점수[i]})); const m={l:35,r:10,t:10,b:26},pw=w-m.l-m.r,ph=h-m.t-m.b,x0=pts[0].t,x1=pts.at(-1).t,yMax=Math.max(3.4,...pts.map(p=>p.v))*1.08; const sx=t=>m.l+(t-x0)/(x1-x0||1)*pw,sy=v=>m.t+(yMax-v)/yMax*ph;
  ctx.clearRect(0,0,w,h);ctx.fillStyle="#fff";ctx.fillRect(0,0,w,h);ctx.font="9px ui-monospace,monospace"; for(let i=0;i<4;i++){const v=yMax*i/3,y=sy(v);ctx.strokeStyle="#e5e9ee";ctx.beginPath();ctx.moveTo(m.l,y);ctx.lineTo(w-m.r,y);ctx.stroke();ctx.fillStyle="#778397";ctx.fillText(fmt(v,1),2,y+3);} linePath(ctx,[[m.l,sy(2)],[w-m.r,sy(2)]],"#cc8b16",1.2,[5,4]);linePath(ctx,[[m.l,sy(3)],[w-m.r,sy(3)]],"#c43d4a",1.2,[5,4]);linePath(ctx,pts.map(p=>[sx(p.t),sy(p.v)]),"#2767c8",1.8);ctx.strokeStyle="#adb8c6";ctx.strokeRect(m.l,m.t,pw,ph);
}

function selectSensor(sensor) { state.activeSensor=sensor; document.querySelectorAll("[data-sensor]").forEach(button=>button.classList.toggle("active",button.dataset.sensor===sensor)); $("overviewSensor").value=sensor; $("automlSensor").value=sensor; renderOverview();renderSensorDetail();renderAutoML();requestAnimationFrame(()=>drawSensorChart($("overviewChart"),sensor)); }
function switchTab(tab) { document.querySelectorAll(".tab").forEach(btn=>btn.classList.toggle("active",btn.dataset.tab===tab));document.querySelectorAll(".tab-panel").forEach(panel=>panel.classList.toggle("active",panel.id===`tab-${tab}`));requestAnimationFrame(()=>{if(tab==="overview"){drawHealthChart();drawSensorChart($("overviewChart"),state.activeSensor);}if(tab==="sensor")drawSensorChart($("sensorChart"),state.activeSensor);}); }
function downloadJson(){if(!state.data)return;const blob=new Blob([JSON.stringify(state.data.요약,null,2)],{type:"application/json"});const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download="rul_summary.json";link.click();URL.revokeObjectURL(link.href);}

$("fileInput").addEventListener("change",event=>setFile(event.target.files[0]));
const drop=$("dropZone");["dragenter","dragover"].forEach(type=>drop.addEventListener(type,e=>{e.preventDefault();drop.classList.add("drag")}));["dragleave","drop"].forEach(type=>drop.addEventListener(type,e=>{e.preventDefault();drop.classList.remove("drag")}));drop.addEventListener("drop",e=>setFile(e.dataTransfer.files[0]));
document.querySelectorAll("[data-sample]").forEach(button=>button.addEventListener("click",()=>loadSample(button.dataset.sample)));
$("analyzeButton").addEventListener("click",analyze);$("downloadButton").addEventListener("click",downloadJson);
document.querySelectorAll(".tab").forEach(button=>button.addEventListener("click",()=>switchTab(button.dataset.tab)));
$("overviewSensor").addEventListener("change",e=>selectSensor(e.target.value));$("automlSensor").addEventListener("change",renderAutoML);
$("sensorButtons").addEventListener("click",e=>{const button=e.target.closest("[data-sensor]");if(button)selectSensor(button.dataset.sensor)});
$("sensorLegend").addEventListener("click",e=>{const button=e.target.closest("[data-series]");if(!button)return;state.visibility[button.dataset.series]=!state.visibility[button.dataset.series];renderSensorDetail();});
window.addEventListener("resize",()=>{if(!state.data)return;clearTimeout(window.__resizeTimer);window.__resizeTimer=setTimeout(()=>{drawHealthChart();drawSensorChart($("overviewChart"),state.activeSensor);drawSensorChart($("sensorChart"),state.activeSensor)},120)});
fetch("/api/health").then(r=>r.json()).then(()=>{$("serverDot").classList.add("online");$("serverText").textContent="분석 서비스 정상"}).catch(()=>{$("serverText").textContent="분석 서비스 연결 실패"});
