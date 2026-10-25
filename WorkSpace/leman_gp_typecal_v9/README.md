# v9 type-specific score calibration

CPU-only experiment using the audited v6 candidate cache. No neural parameters are updated.
Keep six v6 features and fit-only type intercepts. Add GP and boundary-logprob slope deviations for types with at least 500 fitting gold mentions. Ridge is 10 for existing terms and 50 for slope deviations; no search over these constants.

Full and opposite-half fits select v6 or type_slopes using fitting Micro, Macro, baseline preference. Adoption requires no full-dev decrease, pooled crossfit improvement, and no decrease in either held-out half. The halves and underlying checkpoint remain unchanged. No test file is read.

Server: bash run_all.sh
