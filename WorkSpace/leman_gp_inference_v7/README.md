# v7 expanded-candidate inference

Use the existing v4 and parent checkpoints. `bash run_all.sh` performs four-GPU inference followed by CPU logistic/threshold calibration. No optimizer or training loop runs. All measurements are exact original character-span Micro/Macro13 on dev2. Candidate extraction uses scaled GP > -32 instead of > -4. The baseline v6 files remain unchanged.

The route for each held-out half is selected only using its opposite half. Model checkpoints and the overall experiment sequence have already used dev; this remains an exploratory development result.
