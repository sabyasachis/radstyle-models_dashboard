# RadStyle report comparison dashboard

Public, read-only comparison of Qwen3.5-2B outputs with reference Findings and Impressions.

**Site:** https://sabyasachis.github.io/radstyle-models_dashboard/

## Experiment tabs

**Persona-token conditioning:** [open the new tab](https://sabyasachis.github.io/radstyle-models_dashboard/?experiment=persona).
It shows saved report generations, not a live model endpoint. The same 858 joint
and 829 disjoint test reports now have five matched-decoding outputs each
(8,435 outputs): frozen base, joint/disjoint conditioned models, and their
joint/disjoint unconditioned baselines. All four SFT models use final epoch 3
(joint step 923; disjoint step 927), not validation-selected checkpoints.

This tab uses non-thinking text decoding: temperature 1, top-p 1, top-k 20,
presence penalty 2, min-p 0, repetition penalty 1, maximum 576 Impression tokens.
It never mixes outputs from the original tab's decoding profile. Per-report
scores still show three-judge sample SD, **not a CI**;
[aggregate confidence intervals and experimental setup](https://github.com/sabyasachis/radstyle-models/blob/main/artifacts/persona_conditioned_20260922/TEAM_PRESENTATION_HANDOFF_WITH_CIS.md)
are linked separately.

Joint and disjoint have **distinct persona catalogs**. Conditioned cards show
the actual input token: a dedicated token for own-dataset ranks 1-40, or
`<|persona_unknown|>` for held-out ranks and every cross-dataset report.
Cards distinguish dataset ID/OOD from actual seen/unseen persona exposure;
the sidebar and report badge continue to classify the **test catalog**.
Cross-dataset generation is both dataset-OOD and unseen-persona, even for
catalog-ID reports. Matching token strings across models do not mean matching
semantic personas. Token-control ablations are not included.

Tab selection is included in shareable URLs and browser history. Filters,
report selection, navigation and per-tab model selection work in both tabs.
Original public report IDs are retained so the same case stays selected across
tabs, including historical IDs that use the old dataset spelling.

## Original experiment tab

- 858 joined and 829 disjoint test reports: **1,687 reports, 6,748 model outputs**.
- Untouched non-thinking and thinking baselines.
- Joined non-thinking SFT checkpoint-300 and disjoint non-thinking SFT checkpoint-600.
- Findings, reference Impression, generated Impressions, and per-report GREEN/style means and three-judge sample SDs, ROUGE-L F1, deterministic-check result, and token counts.
- Dataset, source, persona rank, ID/OOD and free-text filters; report navigation; model selection; shareable case URLs.

**Persona grouping follows the test catalog**, as requested: ranks 1-40 are ID, ranks 41-50 are OOD. A cross-dataset model does **not** turn an ID report into an OOD report in this dashboard. The model card separately says "Cross-dataset." Untouched-model pretraining exposure is unknown.

Predictions in the original tab belong to the original experimental decoding profile, not Qwen's recommended text recipe. Non-thinking uses temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5; thinking uses 0.6, 0.95, 20, 1.5. The persona tab uses its own matching text-profile base and SFT outputs. Thinking-SFT models are intentionally excluded.

Scores are automated Impression-to-Impression research measurements, not clinical validation. The three judge calls use the same GPT-5.6-Sol model with high reasoning effort. SD is **within-report judge disagreement**, not a confidence interval.

## Public data release

The owner explicitly confirmed on September 21, 2026 that the records are non-identifying and approved/licensed for public release, and that the data derives from open datasets. See [DATA_NOTICE.md](DATA_NOTICE.md). Publicly accessible does not mean upstream license restrictions disappear.

The exporter uses a strict field allowlist. It does not publish raw scratchpads, raw judge responses, original example/source/patient identifiers, persona policy contents, private filesystem paths, or credentials. Public case IDs are ordinal display identifiers. Clinical text is included, publicly downloadable, and rendered with `textContent`, never interpreted as HTML.

## Local development

No runtime dependencies, build step, external fonts, analytics, or API keys are required.

```bash
npm run serve
# Open http://127.0.0.1:4173/
npm test
```

Browser tests:

```bash
npm ci
npx playwright install chromium
npm run test:browser
```

## Re-export approved records

Use the original experiment's private artifact directory as input. Never copy that directory directly into this repository.

```bash
python3 scripts/export_data.py \
  --artifacts /path/to/approved/experiment/artifacts \
  --output docs/data \
  --public-release-approved
```

The exporter verifies test membership, matching example/reference identities across all four models, 50-persona coverage, exact ID/OOD counts, fixed decoding, and complete three-judge scores matching the frozen snapshot. Exported datasets are `docs/data/joint.json` and `disjoint.json`; `catalog.json` contains release provenance and counts.

### Persona experiment export

```bash
python3 -m scripts.export_persona \
  --run /path/to/approved/persona-conditioned-run \
  --approved-data docs/data \
  --output docs/data/persona \
  --public-release-approved
```

The separate allowlisted exporter reads only the ten completed full-test
headline/base/epoch-3 cells. It validates checkpoint identities, decoding,
tokens, actual exposure, identical reference and semantic-persona identities,
full persona coverage, and all four aggregate metrics. Every reference case
must match the already approved public release; historical data is not
overwritten. Only input-token and exposure annotations are added to the
existing public output fields. Index/summary SHA-256 hashes preserve aggregate
provenance without publishing private storage paths or raw judgments.

## GitHub Pages deployment

The published static site is the **`docs/` directory on `main`**. Enable GitHub Pages with **Deploy from a branch → main → /docs**. `.nojekyll` disables Jekyll processing. All asset/data paths are relative so the project-site prefix works.

Keep data releases/profile identities explicit; do not mix predictions from different decoding profiles in a four-model comparison.
