import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {conditioningContext, filterCases, modelContext, personaGroup, readState, shareUrl, validateCatalog, validateDataset} from "../docs/core.js";

test("persona partition depends on catalog rank, not training dataset", () => {
  assert.equal(personaGroup(1), "id");
  assert.equal(personaGroup(40), "id");
  assert.equal(personaGroup(41), "ood");
  assert.equal(personaGroup(50), "ood");
  for (const bad of [0, 51, 1.5, "40", null]) assert.throws(() => personaGroup(bad));
  assert.equal(modelContext({trainingDataset: "disjoint"}, "joint"), "Cross-dataset");
  assert.equal(personaGroup(10), "id");
});

test("persona links preserve the experiment but omit searches and unrelated filters", () => {
  assert.equal(readState("?experiment=persona").experiment, "persona");
  assert.equal(readState("?experiment=invalid").experiment, "original");
  const url = shareUrl("https://example.org/repo/?search=secret&group=ood", {dataset: "joint", experiment: "persona"}, "historical-0042");
  assert.equal(url.search, "?experiment=persona&dataset=joint&case=historical-0042");
});

test("conditioning is dataset scoped, not just catalog-rank scoped", () => {
  const model = {trainingDataset: "joint", conditioned: true};
  assert.deepEqual(conditioningContext(model, "joint", 40), {
    conditioningToken: "<|persona_040|>", datasetExposure: "in_distribution", personaExposure: "in_distribution",
  });
  assert.deepEqual(conditioningContext(model, "joint", 41), {
    conditioningToken: "<|persona_unknown|>", datasetExposure: "in_distribution", personaExposure: "ood",
  });
  assert.deepEqual(conditioningContext(model, "disjoint", 1), {
    conditioningToken: "<|persona_unknown|>", datasetExposure: "ood", personaExposure: "ood",
  });
  assert.equal(conditioningContext({...model, conditioned: false}, "joint", 1).conditioningToken, null);
});

test("query parameters are bounded and share links omit free-text searches", () => {
  assert.equal(readState("?dataset=unknown&persona=999&group=bad").dataset, "joint");
  assert.equal(readState("?persona=41&group=ood").persona, "41");
  const url = shareUrl("https://example.org/repo/?search=secret", {dataset: "disjoint"}, "disjoint-0001");
  assert.equal(url.search, "?dataset=disjoint&case=disjoint-0001");
});

for (const [dataset, total, id, ood] of [["joint", 858, 683, 175], ["disjoint", 829, 675, 154]]) {
  test(`${dataset}: exported coverage, labels, text, scores, and all four models`, () => {
    const data = validateDataset(JSON.parse(readFileSync(new URL(`../docs/data/${dataset}.json`, import.meta.url))), dataset);
    assert.equal(data.cases.length, total);
    assert.equal(filterCases(data.cases, {partition: "id"}).length, id);
    assert.equal(filterCases(data.cases, {partition: "ood"}).length, ood);
    assert.equal(new Set(data.cases.map(c => c.personaRank)).size, 50);
    assert.deepEqual(filterCases(data.cases, {search: "a-no-match-token-987654321"}), []);
    const item = data.cases[0];
    assert.ok(filterCases(data.cases, {search: item.id.toUpperCase()}).some(c => c.id === item.id));
    assert.ok(filterCases(data.cases, {persona: String(item.personaRank)}).every(c => c.personaRank === item.personaRank));
    for (const record of data.cases) {
      assert.deepEqual(Object.keys(record).sort(), ["id", "source", "personaRank", "personaGroup", "findings", "reference", "referenceTokens", "outputs"].sort());
      for (const output of Object.values(record.outputs)) {
        assert.deepEqual(Object.keys(output).sort(), ["text", "tokens", "rougeL", "checksPassed", "truncated", "thinkingFallback", "green", "style"].sort());
      }

    }
    const bad = structuredClone(data);
    bad.cases[0].personaGroup = "invalid";
    assert.throws(() => validateDataset(bad, dataset));
  });
}

for (const [dataset, total, id, ood] of [["joint", 858, 683, 175], ["disjoint", 829, 675, 154]]) {
  test(`${dataset}: conditioned release uses the approved cohort and five epoch-3/base models`, () => {
    const data = validateDataset(JSON.parse(readFileSync(new URL(`../docs/data/persona/${dataset}.json`, import.meta.url))), dataset, "persona");
    const original = JSON.parse(readFileSync(new URL(`../docs/data/${dataset}.json`, import.meta.url)));
    const catalog = validateCatalog(JSON.parse(readFileSync(new URL("../docs/data/persona/catalog.json", import.meta.url))), "persona");
    assert.equal(data.cases.length, total);
    assert.equal(filterCases(data.cases, {partition: "id"}).length, id);
    assert.equal(filterCases(data.cases, {partition: "ood"}).length, ood);
    assert.deepEqual(data.models, catalog.models);
    assert.equal(catalog.totalReports, 1687);
    assert.throws(() => validateDataset(data, dataset, "original"));
    for (const [index, item] of data.cases.entries()) {
      const {outputs, ...reference} = item;
      const {outputs: ignored, ...previous} = original.cases[index];
      assert.deepEqual(reference, previous);
      assert.equal(Object.keys(outputs).length, 5);
      for (const output of Object.values(outputs)) {
        assert.deepEqual(Object.keys(output).sort(), ["text", "tokens", "rougeL", "checksPassed", "truncated",
          "thinkingFallback", "green", "style", "conditioningToken", "datasetExposure", "personaExposure"].sort());
      }
    }
    const bad = structuredClone(data);
    bad.cases[0].outputs[`${dataset}-conditioned`].conditioningToken = "<|persona_999|>";
    assert.throws(() => validateDataset(bad, dataset, "persona"), /token or exposure/);
    const wrongCheckpoint = structuredClone(data);
    wrongCheckpoint.models.find(m => m.id === "joint-conditioned").checkpoint = 300;
    assert.throws(() => validateDataset(wrongCheckpoint, dataset, "persona"), /checkpoint/);
  });
}
