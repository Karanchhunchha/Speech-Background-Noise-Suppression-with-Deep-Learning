# Submission checklist — Project #193

Everything below is checked against the repository as committed; the
reviewer-facing self-check is `verify_submission()` (one command).

## Repository (MathWorks wiki: Submission Instructions & Project Repository Guidelines)

- [x] Clear folder structure; README with description, setup, how to
      run, results, dependencies, contact
- [x] One main entry point: `main.m` (no code edits needed to run)
- [x] Sample input data committed (`data/sample/`); full dataset via
      `download_data()` with pinned revision + SHA-256 verification
- [x] Trained model committed (`models/denoiseNet.mat`) and loaded
      directly by the main script
- [x] Dependencies listed in README; machine-generated list in
      `reports/required_products.txt`
- [x] Tests with expected outputs (`run_all_tests()`); reviewer
      self-check `verify_submission()`
- [x] Documentation: README, `EVALUATION.md`, `ARCHITECTURE.md`,
      `DATASET.md`, `REPRODUCIBILITY.md`, code comments throughout
- [x] Relative paths only (scanned by `verify_submission`); runs from
      a fresh clone (verified in a clean MATLAB Online clone)
- [x] MIT `LICENSE`; `CITATION.cff`

## Results integrity

- [x] STOI implementation cross-checked against the reference
      (`pystoi`, agreement within 0.0005)
- [x] STFT cross-checked against Signal Processing Toolbox `stft`
      (`tests/test_ToolboxCrossCheck.m`, runs when the toolbox exists)
- [x] Published metrics reproduce from the shipped model
      (`verify_submission` step 3, 8-file check within tolerance)
- [x] Weaknesses stated, not hidden: CNN STOI vs the noisy input, the
      PESQ situation, no per-noise-type labels in the corpus

## Owner-only steps (cannot be automated by the maintainer tooling)

- [ ] Add `ci.yml` to `.github/workflows/ci.yml` via the GitHub web UI
      (the maintenance automation lacks the `workflows` permission) —
      until then the README test badge is static, by design
- [ ] GitHub repository topics + social preview image
- [ ] Publish the v1.0.0 release from `RELEASE_NOTES.md`
- [ ] Collect listening-test ratings (`tools/make_listening_test.m`),
      then run `tools/analyze_listening_test.m`
- [ ] Hub submission form (registration confirmed 2026-11-27;
      solution deadline 27 March 2027)
