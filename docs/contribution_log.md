# Contribution log (C2)

Owners: **DE** = Alex Rees (Data Engineer, partner). **ML** = Jiaxiao Chen (ML Engineer).

Rows up to 2026-10-11 are Alex Rees's work in `notebooks/data_exploration.ipynb`, cited by verified commit hash.
The "git author" column is what `git log` records for each commit (`AlexRees23`, and `Alexander Rees` for the
initial commit).

Rows dated 2026-10-12 are the ML completion files prepared by Jiaxiao Chen with AI assistance (Claude for
implementation and drafting, Codex for planning and verification). They are included in the completion commit
with the subject "Complete retail recommendation project and individual reflection", authored as Jiaxiao Chen.
The completion commit is identifiable by this subject; use `git log --oneline -- <file>` to verify.

| Date | Task | Owner | Evidence (commit, file or figure) | Git author | Status |
| --- | --- | --- | --- | --- | --- |
| 2026-10-09 | Repository created | DE | `3eb1984` Initial commit (`.gitignore`, `README.md`) | `Alexander Rees` | done |
| 2026-10-09 | Step 0 download; project skeleton; first data exploration | DE | `d2ba10f` (zip, `notebooks/data_exploration.ipynb`, empty `build_matrix.py` / `requirements.txt`, `data/` ignored) | `AlexRees23` | done |
| 2026-10-09 | DE1 data profile; row count in README | DE | `91b282a` "Finish step 1" (DE1 section, README row counts) | `AlexRees23` | done |
| 2026-10-10 | DE2 cleaning log | DE | `e798bb8` "DE2 work" | `AlexRees23` | done |
| 2026-10-10 | DE3 time split, DE4 sparse matrix and index maps | DE | `b15e231` "DE3 and DE4 work" | `AlexRees23` | done |
| 2026-10-11 | DE5 matrix report; exploratory ML1 method note and ML2 (dense SVD, k = 100 chosen) | DE | `7eed7d6` "ML2 work" | `AlexRees23` | done |
| 2026-10-12 | J1 success targets written at the top of the README (completion stage, not Day 1) | ML | `README.md` | Jiaxiao Chen | included in completion commit; see file history |
| 2026-10-12 | DE6 `python build_matrix.py` rebuilds DE1-DE5 from the raw file (reproduces every partner count) | ML | `build_matrix.py`, `reports/{data_profile,sheet_counts,cleaning_log,split_log}.csv`, `reports/matrix_report.json` | Jiaxiao Chen | included in completion commit; see file history |
| 2026-10-12 | R6 pinned requirements, fixed seeds, one-command rebuild; raw archive git-ignored | ML | `requirements.txt`, `run_project.py`, `pytest.ini`, `.gitignore` | Jiaxiao Chen | included in completion commit; see file history |
| 2026-10-12 | ML2 sparse TruncatedSVD sweep (k = 10 to 200), curve, k = 100 kept | ML | `model_analysis.py`, `reports/svd_curve.csv`, `figures/svd_curve.png` | Jiaxiao Chen | included in completion commit; see file history |
| 2026-10-12 | ML3 item-to-item and customer recommendations | ML | `model_analysis.py` (`recommend_similar`, `recommend_for_customer`, `item_neighbors`) | Jiaxiao Chen | included in completion commit; see file history |
| 2026-10-12 | ML4 hit rate and recall at 10 vs best sellers, customer/pair/product coverage, uncertainty, nightly timing | ML | `reports/evaluation.csv`, `reports/results.json`, `figures/evaluation.png` | Jiaxiao Chen | included in completion commit; see file history |
| 2026-10-12 | ML5 PCA vs SVD comparison table | ML | `reports/comparison.csv` | Jiaxiao Chen | included in completion commit; see file history |
| 2026-10-12 | ML6 customer map, supplementary component-score map and four named segments | ML | `figures/customer_map.png`, `figures/customer_map_components.png`, `reports/segments.csv`, `reports/segment_profiles.json`, `docs/segment_analysis.md` | Jiaxiao Chen | included in completion commit; see file history |
| 2026-10-12 | ML7 purchase-value sensitivity | ML | `reports/sensitivity.csv` | Jiaxiao Chen | included in completion commit; see file history |
| 2026-10-12 | Tests on synthetic fixtures (30) | ML | `tests/` | Jiaxiao Chen | included in completion commit; see file history |
| 2026-10-12 | Completion notebook, executed top to bottom | ML | `notebooks/model_analysis.ipynb` | Jiaxiao Chen | included in completion commit; see file history |
| 2026-10-12 | Decision log, contribution log, templates, checklist | ML | `docs/` | Jiaxiao Chen | included in completion commit; see file history |
| 2026-10-12 | J2 memo for Sarah (one page); prepared by Jiaxiao Chen with Claude assistance | ML | `output/pdf/memo.pdf` from `scripts/create_submission.py memo` | Jiaxiao Chen | prepared and included in completion commit; team review and presentation pending |
| 2026-10-12 | J3 slides (six slides, 5-minute speaker notes); prepared by Jiaxiao Chen with Claude assistance; each member presents their own part | ML | `output/slides/presentation.pptx` from `scripts/create_slides.mjs` | Jiaxiao Chen | prepared and included in completion commit; team review and presentation pending |
| (to do) | C3 peer evaluation (confidential, each member separately) | each | `docs/peer_evaluation_template.md` (do not commit filled forms) | - | open |
| 2026-10-12 | Individual reflection draft, Jiaxiao Chen (first-person draft prepared with AI assistance) | ML | `docs/individual_reflection.md`, `output/pdf/individual_reflection.pdf` from `scripts/create_reflection.py` | Jiaxiao Chen | draft included in completion commit; student review pending |
| (to do) | Individual reflection, Alex Rees (half a page, own words) | DE | `docs/individual_reflection_template.md` | - | open |

## Sign-off

To be filled in by hand. Each partner signs and dates once they have checked that the rows above are accurate.

| Member | Signature | Date |
| --- | --- | --- |
| Alex Rees (Data Engineer) | | |
| Jiaxiao Chen (ML Engineer) | | |
