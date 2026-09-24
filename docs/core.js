export const MODEL_IDS = ["base_nonthinking", "base_thinking", "joint_nonthinking", "disjoint_nonthinking"];
export const EXPERIMENTS = {
  original: {directory: "./data", modelIds: MODEL_IDS, profile: "original-experiment"},
  persona: {directory: "./data/persona", modelIds: ["base", "joint-unconditioned", "joint-conditioned", "disjoint-unconditioned", "disjoint-conditioned"], profile: "qwen-text-nonthinking"},
};
export const SOURCES = {ct_rate: "CT-RATE", mr_rate: "MR-RATE", rexgradient: "REXGradient"};

export function personaGroup(rank) {
  if (!Number.isInteger(rank) || rank < 1 || rank > 50) throw new Error("Invalid persona rank");
  return rank <= 40 ? "id" : "ood";
}

export function modelContext(model, dataset) {
  if (model.trainingDataset === null) return "Untouched baseline";
  return model.trainingDataset === dataset ? "Matched dataset" : "Cross-dataset";
}

export function conditioningContext(model, dataset, rank) {
  personaGroup(rank);
  const baseline = model.trainingDataset === null;
  const datasetId = model.trainingDataset === dataset;
  const personaId = datasetId && rank <= 40;
  return {
    conditioningToken: model.conditioned ? (personaId ? `<|persona_${String(rank).padStart(3, "0")}|>` : "<|persona_unknown|>") : null,
    datasetExposure: baseline ? "not_applicable" : datasetId ? "in_distribution" : "ood",
    personaExposure: baseline ? "not_applicable" : personaId ? "in_distribution" : "ood",
  };
}

export function filterCases(cases, {partition = "all", persona = "all", source = "all", search = ""}) {
  const query = search.trim().toLocaleLowerCase();
  return cases.filter(item =>
    (partition === "all" || personaGroup(item.personaRank) === partition) &&
    (persona === "all" || item.personaRank === Number(persona)) &&
    (source === "all" || item.source === source) &&
    (!query || [item.id, item.findings, item.reference, ...Object.values(item.outputs).map(v => v.text)]
      .some(text => text.toLocaleLowerCase().includes(query)))
  );
}

export function readState(search) {
  const p = new URLSearchParams(search);
  const rank = Number(p.get("persona"));
  return {
    experiment: p.get("experiment") === "persona" ? "persona" : "original",
    dataset: p.get("dataset") === "disjoint" ? "disjoint" : "joint",
    partition: ["id", "ood"].includes(p.get("group")) ? p.get("group") : "all",
    persona: Number.isInteger(rank) && rank >= 1 && rank <= 50 ? String(rank) : "all",
    source: Object.hasOwn(SOURCES, p.get("source")) ? p.get("source") : "all",
    caseId: p.get("case") || null,
    search: "",
  };
}

export function shareUrl(base, state, caseId) {
  const url = new URL(base);
  url.search = "";
  url.hash = "";
  if (state.experiment === "persona") url.searchParams.set("experiment", "persona");
  url.searchParams.set("dataset", state.dataset);
  url.searchParams.set("case", caseId);
  return url;
}

function validateModels(models, experiment) {
  const ids = EXPERIMENTS[experiment].modelIds;
  if (!Array.isArray(models) || models.length !== ids.length ||
      ids.some(id => !models.some(m => m.id === id))) throw new Error("Model catalog is incomplete");
  if (experiment === "persona") {
    for (const model of models) {
      const training = model.id === "base" ? null : model.id.startsWith("joint-") ? "joint" : "disjoint";
      if (model.trainingDataset !== training || model.epoch !== (training ? 3 : 0) ||
          model.checkpoint !== (training === null ? null : training === "joint" ? 923 : 927) ||
          model.conditioned !== model.id.endsWith("-conditioned") || model.mode !== "nonthinking") {
        throw new Error("Invalid final-checkpoint model metadata");
      }
    }
  }
}

export function validateCatalog(catalog, experiment) {
  if (catalog.schemaVersion !== 1 || catalog.decodingProfile !== EXPERIMENTS[experiment].profile ||
      !Number.isInteger(catalog.totalReports) || catalog.totalReports < 1) throw new Error("Invalid report catalog");
  validateModels(catalog.models, experiment);
  const date = experiment === "persona" ? catalog.exportedAt : catalog.snapshotCompletedAt;
  if (typeof date !== "string" || !Number.isFinite(Date.parse(date))) throw new Error("Invalid catalog date");
  return catalog;
}

export function validateDataset(data, dataset, experiment = "original") {
  const number = value => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
  if (data.schemaVersion !== 1 || data.dataset !== dataset || data.decodingProfile !== EXPERIMENTS[experiment].profile ||
      !Array.isArray(data.cases) || !data.cases.length) {
    throw new Error("Invalid dataset file");
  }
  validateModels(data.models, experiment);
  const ids = new Set();
  for (const item of data.cases) {
    if (typeof item.id !== "string" || ids.has(item.id)) throw new Error("Duplicate or invalid public report ID");
    ids.add(item.id);
    if (!Object.hasOwn(SOURCES, item.source) || item.personaGroup !== personaGroup(item.personaRank)) {
      throw new Error("Invalid source or persona classification");
    }
    if (typeof item.findings !== "string" || !item.findings.trim() ||
        typeof item.reference !== "string" || !item.reference.trim()) throw new Error("Missing reference text");
    for (const model of data.models) {
      const v = item.outputs?.[model.id];
      if (!v || typeof v.text !== "string" || !v.text.trim() ||
          !number(v.green?.mean) || !number(v.green?.sd) ||
          !number(v.style?.mean) || !number(v.style?.sd) || !number(v.rougeL) ||
          typeof v.checksPassed !== "boolean" || !Number.isInteger(v.tokens) || v.tokens < 0 ||
          typeof v.truncated !== "boolean" || typeof v.thinkingFallback !== "boolean") throw new Error("Missing or invalid model output");
      if (experiment === "persona") {
        const expected = conditioningContext(model, dataset, item.personaRank);
        if (Object.entries(expected).some(([key, value]) => v[key] !== value)) {
          throw new Error("Incorrect persona token or exposure annotation");
        }
      }
    }
  }
  return data;
}
