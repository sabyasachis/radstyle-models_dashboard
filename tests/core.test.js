import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {filterCases, modelContext, personaGroup, readState, shareUrl, validateDataset} from "../docs/core.js";

test("persona partition depends on catalog rank, not training dataset", () => {
  assert.equal(personaGroup(1), "id");
  assert.equal(personaGroup(40), "id");
  assert.equal(personaGroup(41), "ood");
  assert.equal(personaGroup(50), "ood");
  for (const bad of [0, 51, 1.5, "40", null]) assert.throws(() => personaGroup(bad));
  assert.equal(modelContext({trainingDataset: "disjoint"}, "joint"), "Cross-dataset");
  assert.equal(personaGroup(10), "id");
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
