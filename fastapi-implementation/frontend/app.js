

const API = {
  health:    "/api/health",
  recommend: "/api/recommend",
};

const BATCH_SIZE = 5;       
const STAGGER_MS = 120;

const $form          = document.getElementById("upload-form");
const $dropzone      = document.getElementById("dropzone");
const $fileInput     = document.getElementById("file-input");
const $dropzoneHint  = document.getElementById("dropzone-hint");
const $submitBtn     = document.getElementById("submit-btn");
const $statusBar     = document.getElementById("status-bar");
const $results       = document.getElementById("results");
const $summary       = document.getElementById("summary");
const $favsGrid      = document.getElementById("favorites-grid");
const $recsGrid      = document.getElementById("recommendations-grid");
const $revealMore    = document.getElementById("reveal-more");
const $healthInfo    = document.getElementById("health-info");
const $topK          = document.getElementById("top-k");
const $useFinetuning = document.getElementById("use-finetuning");
const $useQuadratic  = document.getElementById("use-quadratic");
const $useAnti       = document.getElementById("use-anti");


let pendingRecs = [];
let revealedCount = 0;


let currentFile = null;


function setStatus(msg, kind = "info") {
  $statusBar.hidden = !msg;
  $statusBar.textContent = msg || "";
  $statusBar.className = "status-bar " + kind;
}

function setSubmitEnabled(enabled) { $submitBtn.disabled = !enabled; }

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function ratingStars(ratingNorm) {
  const stars = Math.round(((ratingNorm * (5.0 - 0.5)) + 0.5) * 2) / 2;
  return "★".repeat(Math.floor(stars)) + (stars % 1 ? "½" : "");
}


async function loadHealth() {
  try {
    const res = await fetch(API.health);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const j = await res.json();
    const m = j.metrics || {};
    const p10 = m.precision_at_k != null ? m.precision_at_k.toFixed(4) : "—";
    const r10 = m.recall_at_k    != null ? m.recall_at_k.toFixed(4)    : "—";
    $healthInfo.textContent =
      `Model v${j.version} · ${j.n_users} users · ${j.n_movies} movies · P@10 ${p10} · R@10 ${r10}`;
  } catch (e) {
    $healthInfo.innerHTML = `<span style="color:var(--red)">Model unavailable — ${escapeHtml(e.message)}</span>`;
  }
}
loadHealth();


["dragenter", "dragover"].forEach(ev =>
  $dropzone.addEventListener(ev, e => {
    e.preventDefault(); e.stopPropagation();
    $dropzone.classList.add("dragging");
  })
);
["dragleave", "drop"].forEach(ev =>
  $dropzone.addEventListener(ev, e => {
    e.preventDefault(); e.stopPropagation();
    $dropzone.classList.remove("dragging");
  })
);
$dropzone.addEventListener("drop", e => {
  if (e.dataTransfer && e.dataTransfer.files.length) {
    onFileChosen(e.dataTransfer.files[0]);
  }
});
$fileInput.addEventListener("change", () => {
  onFileChosen($fileInput.files && $fileInput.files[0]);
});

function onFileChosen(file) {
  if (file) {
    currentFile = file;
    $dropzone.classList.add("has-file");
    $dropzoneHint.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
    setSubmitEnabled(true);
    setStatus("");
  } else {
    currentFile = null;
    $dropzone.classList.remove("has-file");
    $dropzoneHint.textContent = "or click to select it";
    setSubmitEnabled(false);
  }
}


$form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = currentFile;
  if (!f) { setStatus("Select a .zip file first.", "error"); return; }

  setSubmitEnabled(false);
  setStatus("Processing your history and building your profile…", "loading");
  $results.hidden = true;

  const formData = new FormData();
  formData.append("file", f);
  formData.append("top_k", $topK.value);
  formData.append("use_finetuning", $useFinetuning.checked ? "true" : "false");
  formData.append("use_quadratic",  $useQuadratic.checked  ? "true" : "false");
  formData.append("use_anti",       $useAnti.checked       ? "true" : "false");

  try {
    const t0 = performance.now();
    const res = await fetch(API.recommend, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    const dt = ((performance.now() - t0) / 1000).toFixed(2);
    setStatus(`Done in ${dt}s.`, "info");
    renderResults(data);
  } catch (e) {
    setStatus(`Error: ${e.message}`, "error");
  } finally {
    currentFile = null;
    $fileInput.value = "";
    onFileChosen(null);
  }
});

function renderResults(data) {
  $results.hidden = false;
  renderSummary(data);
  renderFavorites(data.favorites || []);
  setupRecommendations(data.recommendations || []);
  setTimeout(() => $results.scrollIntoView({ behavior: "smooth", block: "start" }), 80);
}

function renderSummary(d) {
  const cells = [
    { label: "User",       value: d.username ? escapeHtml(d.username) : "<small>unnamed</small>" },
    { label: "Scenario",   value: `<span class="scenario-badge ${d.scenario}">${d.scenario}</span>` },
    { label: "Ratings",    value: `${d.n_total_ratings}` },
    { label: "In catalog", value: `${d.n_in_catalog}<small> / ${d.n_in_catalog + d.n_out_catalog}</small>` },
    { label: "Positives",  value: `${d.n_positives_used}` },
    { label: "Fold-in",    value: d.fine_tuning_used ? `Yes<small> (${d.finetune_losses.length})</small>` : "No" },
  ];
  $summary.innerHTML = cells.map(c =>
    `<div class="summary-cell"><div class="label">${escapeHtml(c.label)}</div><div class="value">${c.value}</div></div>`
  ).join("");
}


function posterCardHTML(it, { isFavorite = false } = {}) {
  const title = escapeHtml(it.title || it.title_normalized);
  const year  = it.year != null ? escapeHtml(String(parseInt(it.year, 10))) : "";

  let media;
  const fallbackHTML = `<div class="poster-fallback"><div class="fb-title">${title}</div>${year ? `<div class="fb-year">${year}</div>` : ""}</div>`;
  if (it.poster_url) {
    media = `<img class="poster-img" src="${escapeHtml(it.poster_url)}" alt="${title}" loading="lazy"
                  onerror="this.outerHTML='${fallbackHTML.replace(/'/g, "\\'")}'" />`;
  } else {
    media = fallbackHTML;
  }

  const rankLabel = isFavorite ? "♥" : `#${it.rank}`;

  const overlayBottom = isFavorite
    ? `<div class="poster-stars">${ratingStars(it.rating_norm)}</div>`
    : "";

  const linkHint = it.letterboxd_url
    ? `<div class="poster-link-hint">View on Letterboxd →</div>` : "";

  const inner = `
    ${media}
    <div class="poster-rank">${rankLabel}</div>
    <div class="poster-overlay">
      <div class="poster-title">${title}</div>
      <div class="poster-meta">${year}${it.director ? " · " + escapeHtml(it.director) : ""}</div>
      ${overlayBottom}
      ${linkHint}
    </div>`;

  if (it.letterboxd_url) {
    return `<a class="poster-card${isFavorite ? " is-favorite" : ""}" href="${escapeHtml(it.letterboxd_url)}" target="_blank" rel="noopener">${inner}</a>`;
  }
  return `<div class="poster-card${isFavorite ? " is-favorite" : ""}">${inner}</div>`;
}

function renderFavorites(items) {
  if (!items.length) {
    $favsGrid.innerHTML = `<p style="color:var(--paper-faint)">No favorites detected in the catalog.</p>`;
    return;
  }
  $favsGrid.innerHTML = items.map(it => posterCardHTML(it, { isFavorite: true })).join("");
  revealCards($favsGrid.querySelectorAll(".poster-card"), 0);
}


function setupRecommendations(items) {
  $recsGrid.innerHTML = "";
  pendingRecs = items.slice();
  revealedCount = 0;
  revealNextBatch();
}

function revealNextBatch() {
  const batch = pendingRecs.slice(revealedCount, revealedCount + BATCH_SIZE);
  if (!batch.length) return;

  const startIndex = revealedCount;
  batch.forEach(it => {
    $recsGrid.insertAdjacentHTML("beforeend", posterCardHTML(it));
  });
  revealedCount += batch.length;

  const allCards = $recsGrid.querySelectorAll(".poster-card");
  const newCards = Array.from(allCards).slice(startIndex);
  revealCards(newCards, 0);

  if (revealedCount < pendingRecs.length) {
    $revealMore.hidden = false;
    $revealMore.textContent = `Show ${Math.min(BATCH_SIZE, pendingRecs.length - revealedCount)} more`;
  } else {
    $revealMore.hidden = true;
  }
}

function revealCards(cards, baseDelay) {
  cards.forEach((card, i) => {
    setTimeout(() => card.classList.add("revealed"), baseDelay + i * STAGGER_MS);
  });
}

$revealMore.addEventListener("click", () => {
  revealNextBatch();
});
