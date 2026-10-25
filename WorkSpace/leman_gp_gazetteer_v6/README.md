# LeMan-GP v6: train-only entity gazetteer calibration

Reuse the completed v4 epoch-1 EMA weights and v5 single-model calibration as the fixed baseline. Add only two span features: the log counts of an exact surface with the candidate type and with another type in original train2 gold mentions. The calibrator learns their contribution; dictionary entries never force an entity prediction.

`bash run.sh` builds the dictionary from train2.txt and runs CPU calibration on existing returned GP/boundary caches. It fits each scaler, logistic model, eligible-type set and decoder only on the fitting partition. The two text-hash halves are unchanged from v5. Feature-family selection is also made on the fitting partition. Completed fits can be reused after an interruption.

Adopt v6 only if full-dev Micro does not decrease, combined crossfit Micro increases and neither held-out half decreases. Otherwise selection.json retains the exact v5 model and calibrator.

`bash predict.sh --split dev` applies the frozen selected calibrator after an actual four-GPU forward. No neural model training is required. Test is read only if explicitly requested with --split test.

Server root: /root/data/WorkSpace. Output: output/leman_gazetteer_v6_seed42. Existing v4 best.pt and lexicon.json remain the model inputs. Preserve the new selection.json and train_gazetteer.json together for inference.

Requirements are unchanged: Python with torch, transformers, numpy and the existing v4 dependencies. No installation or model download is needed in the verified server environment.

All 16 frozen v4 Python sources and calibration_core.py are copied unchanged. The neural forward graph, loss, model weights and lexicon used by the neural model are unchanged. This is exploratory development-set optimization on the same previously selected checkpoint.
