export const MODEL_IDS = ["base_nonthinking", "base_thinking", "joint_nonthinking", "disjoint_nonthinking"];
export const SOURCES = {ct_rate: "CT-RATE", mr_rate: "MR-RATE", rexgradient: "REXGradient"};

export function personaGroup(rank) {
  if (!Number.isInteger(rank) || rank < 1 || rank > 50) throw new Error("Invalid persona rank");
  return rank <= 40 ? "id" : "ood";
}

export function modelContext(model, dataset) {
  if (model.trainingDataset === null) return "Untouched baseline";
  return model.trainingDataset === dataset ? "Matched dataset" : "Cross-dataset";
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
  url.searchParams.set("dataset", state.dataset);
  url.searchParams.set("case", caseId);
  return url;
}

export function validateDataset(data, dataset) {
  const number = value => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
  if (data.schemaVersion !== 1 || data.dataset !== dataset || !Array.isArray(data.cases) || !data.cases.length) {
    throw new Error("Invalid dataset file");
  }
  if (!Array.isArray(data.models) || data.models.length !== MODEL_IDS.length ||
      MODEL_IDS.some(id => !data.models.some(m => m.id === id))) throw new Error("Model catalog is incomplete");
  const ids = new Set();
  for (const item of data.cases) {
    if (typeof item.id !== "string" || ids.has(item.id)) throw new Error("Duplicate or invalid public report ID");
    ids.add(item.id);
    if (!Object.hasOwn(SOURCES, item.source) || item.personaGroup !== personaGroup(item.personaRank)) {
      throw new Error("Invalid source or persona classification");
    }
    if (typeof item.findings !== "string" || !item.findings.trim() ||
        typeof item.reference !== "string" || !item.reference.trim()) throw new Error("Missing reference text");
    for (const id of MODEL_IDS) {
      const v = item.outputs?.[id];
      if (!v || typeof v.text !== "string" || !v.text.trim() ||
          !number(v.green?.mean) || !number(v.green?.sd) ||
          !number(v.style?.mean) || !number(v.style?.sd) || !number(v.rougeL) ||
          typeof v.checksPassed !== "boolean") throw new Error("Missing or invalid model output");
    }
  }
  return data;
}
