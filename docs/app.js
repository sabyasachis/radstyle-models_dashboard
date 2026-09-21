import {MODEL_IDS, SOURCES, filterCases, modelContext, personaGroup, readState, shareUrl, validateDataset} from "./core.js";

const $ = id => document.getElementById(id);
const state = {...readState(location.search), page: 0, selected: null};
const pageSize = 25;
const cache = new Map();
const visible = new Set(MODEL_IDS);
let data;
let matches = [];
let requestVersion = 0;
let searchTimer;

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function fail(error) {
  $("error").hidden = false;
  $("error").textContent = `Could not display the reports: ${error.message}. Check your connection and reload the page.`;
  $("status").textContent = "Loading failed.";
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Data request returned HTTP ${response.status}`);
  return response.json();
}

function updateUrl() {
  const url = new URL(location.href);
  url.search = "";
  url.searchParams.set("dataset", state.dataset);
  if (state.partition !== "all") url.searchParams.set("group", state.partition);
  if (state.persona !== "all") url.searchParams.set("persona", state.persona);
  if (state.source !== "all") url.searchParams.set("source", state.source);
  if (state.selected) url.searchParams.set("case", state.selected.id);
  history.replaceState(null, "", url);
}

function syncFilters() {
  document.querySelectorAll('input[name="dataset"]').forEach(input => { input.checked = input.value === state.dataset; });
  $("partition").value = state.partition;
  $("persona").value = state.persona;
  $("source").value = state.source;
  $("search").value = state.search;
}

function selectCase(item) {
  state.selected = item;
  state.caseId = item?.id || null;
  if (item) state.page = Math.floor(matches.indexOf(item) / pageSize);
  renderList();
  renderReport();
  updateUrl();
}

function applyFilters(preferredId) {
  if (!data) return;
  matches = filterCases(data.cases, state);
  const requested = preferredId || state.selected?.id;
  selectCase(matches.find(item => item.id === requested) || matches[0] || null);
  $("status").textContent = `${matches.length.toLocaleString()} matching ${data.label.toLowerCase()} reports.`;
}

function renderList() {
  const pages = Math.max(1, Math.ceil(matches.length / pageSize));
  state.page = Math.max(0, Math.min(state.page, pages - 1));
  $("match-count").textContent = `${matches.length} / ${data.cases.length}`;
  const fragment = document.createDocumentFragment();
  for (const item of matches.slice(state.page * pageSize, (state.page + 1) * pageSize)) {
    const li = node("li");
    const button = node("button", "case-button");
    button.type = "button";
    button.setAttribute("aria-current", String(item.id === state.selected?.id));
    button.append(node("strong", "", item.id));
    button.append(node("span", "case-meta", `${SOURCES[item.source]} · Persona ${item.personaRank} · ${personaGroup(item.personaRank).toUpperCase()}`));
    button.append(node("span", "preview", item.reference));
    button.addEventListener("click", () => selectCase(item));
    li.append(button);
    fragment.append(li);
  }
  $("case-list").replaceChildren(fragment);
  $("page-status").textContent = `Page ${state.page + 1} of ${pages}`;
  $("page-prev").disabled = state.page === 0;
  $("page-next").disabled = state.page + 1 >= pages;
}

function metric(label, value, sd) {
  const cell = node("div");
  cell.append(node("span", "metric-label", label));
  cell.append(node("span", "metric-value", typeof value === "number" ? value.toFixed(3) : value));
  if (sd !== undefined) {
    cell.append(node("span", "metric-sd", `± ${sd.toFixed(3)} judge SD`));
    cell.title = "Mean and sample SD of three judge scores for this fixed output; not a confidence interval.";
  }
  return cell;
}

function renderModels() {
  const item = state.selected;
  const fragment = document.createDocumentFragment();
  if (!item) return;
  for (const model of data.models.filter(m => visible.has(m.id))) {
    const output = item.outputs[model.id];
    const card = node("article", `panel model-card ${model.trainingDataset ? "sft" : ""}`);
    card.dataset.model = model.id;
    const heading = node("div", "card-heading");
    const title = node("div");
    title.append(node("h3", "", model.label));
    title.append(node("p", "model-subtitle", model.checkpoint === null ? "Original Qwen3.5-2B weights" : `Selected checkpoint ${model.checkpoint} · trained on 40 personas`));
    const context = modelContext(model, state.dataset);
    heading.append(title, node("span", `badge ${context === "Cross-dataset" ? "cross" : ""}`, context));
    card.append(heading, node("p", "report-text", output.text));
    const scores = node("div", "metrics");
    scores.append(metric("GREEN", output.green.mean, output.green.sd), metric("Style", output.style.mean, output.style.sd),
      metric("ROUGE-L", output.rougeL), metric("Checks", output.checksPassed ? "Pass" : "Flagged"));
    card.append(scores, node("p", "token-note", `${output.tokens} Impression tokens · 3 GREEN + 3 style judgments`));
    if (output.truncated) card.append(node("p", "warning-note", "Impression reached the output-token limit."));
    if (output.thinkingFallback) card.append(node("p", "warning-note", "Base-thinking scratchpad was invalid after retry; this Impression used empty-thinking fallback."));
    fragment.append(card);
  }
  if (!visible.size) fragment.append(node("p", "empty panel", "Select at least one model above to compare its Impression."));
  $("model-grid").replaceChildren(fragment);
}

function renderReport() {
  const item = state.selected;
  $("empty").hidden = Boolean(item);
  $("case-content").hidden = !item;
  if (!item) return;
  $("case-source").textContent = `${data.label} test set · ${SOURCES[item.source]}`;
  $("case-title").textContent = item.id;
  $("case-tags").replaceChildren(
    node("span", `badge ${item.personaGroup}`, `${item.personaGroup.toUpperCase()} persona`),
    node("span", "badge", `Catalog rank ${item.personaRank} / 50`),
  );
  $("findings").textContent = item.findings;
  $("reference").textContent = item.reference;
  $("reference-tokens").textContent = `${item.referenceTokens} reference tokens`;
  const index = matches.indexOf(item);
  $("case-prev").disabled = index <= 0;
  $("case-next").disabled = index >= matches.length - 1;
  renderModels();
}

async function loadDataset() {
  const version = ++requestVersion;
  const dataset = state.dataset;
  $("error").hidden = true;
  $("workspace").hidden = true;
  $("status").textContent = `Loading ${dataset === "joint" ? "joined" : "disjoint"} reports...`;
  try {
    if (!cache.has(dataset)) cache.set(dataset, validateDataset(await fetchJson(`./data/${dataset}.json`), dataset));
    if (version !== requestVersion) return;
    data = cache.get(dataset);
    state.selected = null;
    state.page = 0;
    syncFilters();
    applyFilters(state.caseId);
    $("workspace").hidden = false;
  } catch (error) {
    if (version === requestVersion) fail(error);
  }
}

for (let rank = 1; rank <= 50; rank++) $("persona").append(new Option(`Persona ${rank}`, String(rank)));
for (const [value, label] of Object.entries(SOURCES)) $("source").append(new Option(label, value));

document.querySelectorAll('input[name="dataset"]').forEach(input => {
  input.addEventListener("change", () => {
    clearTimeout(searchTimer);
    state.dataset = input.value;
    state.caseId = null;
    loadDataset();
  });
});
for (const [id, key] of [["partition", "partition"], ["persona", "persona"], ["source", "source"]]) {
  $(id).addEventListener("change", () => {
    state[key] = $(id).value;
    if (key === "partition" && state.persona !== "all" && state.partition !== "all" &&
        personaGroup(Number(state.persona)) !== state.partition) state.persona = "all";
    if (key === "persona" && state.persona !== "all" && state.partition !== "all" &&
        personaGroup(Number(state.persona)) !== state.partition) state.partition = "all";
    syncFilters();
    applyFilters();
  });
}
$("search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.search = $("search").value;
    applyFilters();
  }, 180);
});
$("reset").addEventListener("click", () => {
  clearTimeout(searchTimer);
  Object.assign(state, {partition: "all", persona: "all", source: "all", search: ""});
  syncFilters();
  applyFilters();
});
for (const [id, direction] of [["page-prev", -1], ["page-next", 1]]) {
  $(id).addEventListener("click", () => { state.page += direction; renderList(); $("case-list").scrollTop = 0; });
}
for (const [id, direction] of [["case-prev", -1], ["case-next", 1]]) {
  $(id).addEventListener("click", () => {
    const next = matches[matches.indexOf(state.selected) + direction];
    if (next) selectCase(next);
  });
}
$("share").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(shareUrl(location.href, state, state.selected.id).href);
    $("status").textContent = "Report link copied. It opens this report without other filters.";
  } catch (error) {
    $("status").textContent = `Clipboard unavailable (${error.name}). Copy the report URL from your address bar instead.`;
  }
});
window.addEventListener("popstate", () => {
  Object.assign(state, readState(location.search));
  loadDataset();
});

async function start() {
  try {
    const catalog = await fetchJson("./data/catalog.json");
    if (catalog.schemaVersion !== 1 || !Array.isArray(catalog.models) ||
        catalog.models.length !== 4 || !Number.isInteger(catalog.totalReports)) throw new Error("Invalid report catalog");
    $("total-reports").textContent = catalog.totalReports.toLocaleString();
    $("snapshot-date").textContent = `Snapshot ${new Date(catalog.snapshotCompletedAt).toISOString().slice(0, 10)} · original decoding`;
    for (const model of catalog.models) {
      const label = node("label", "model-toggle");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = true;
      input.value = model.id;
      input.addEventListener("change", () => {
        if (input.checked) visible.add(model.id); else visible.delete(model.id);
        renderModels();
      });
      label.append(input, node("span", "", model.shortLabel));
      $("model-options").append(label);
    }
    await loadDataset();
  } catch (error) { fail(error); }
}
start();
