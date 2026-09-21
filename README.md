# RadStyle report comparison dashboard

Public, read-only comparison of Qwen3.5-2B outputs with reference Findings and Impressions.

**Site:** https://sabyasachis.github.io/radstyle-models_dashboard/

## Included results

- 858 joined and 829 disjoint test reports: **1,687 reports, 6,748 model outputs**.
- Untouched non-thinking and thinking baselines.
- Joined non-thinking SFT checkpoint-300 and disjoint non-thinking SFT checkpoint-600.
- Findings, reference Impression, generated Impressions, and per-report GREEN/style means and three-judge sample SDs, ROUGE-L F1, deterministic-check result, and token counts.
- Dataset, source, persona rank, ID/OOD and free-text filters; report navigation; model selection; shareable case URLs.

**Persona grouping follows the test catalog**, as requested: ranks 1-40 are ID, ranks 41-50 are OOD. A cross-dataset model does **not** turn an ID report into an OOD report in this dashboard. The model card separately says "Cross-dataset." Untouched-model pretraining exposure is unknown.

All shown predictions belong to the original experimental decoding profile, not Qwen's recommended text recipe. Non-thinking uses temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5; thinking uses 0.6, 0.95, 20, 1.5. The recommended-text follow-up is not included. Thinking-SFT models are intentionally excluded.

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

## GitHub Pages deployment

The published static site is the **`docs/` directory on `main`**. Enable GitHub Pages with **Deploy from a branch → main → /docs**. `.nojekyll` disables Jekyll processing. All asset/data paths are relative so the project-site prefix works.

Keep data releases/profile identities explicit; do not mix predictions from different decoding profiles in a four-model comparison.
