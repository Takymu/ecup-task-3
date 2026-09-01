"""Full refit (incl. val anchor) of the Optuna-tuned two-stage -> *_full models for predict_test.

Reads best params + iters from opt_two_stage.meta.json, trains clf/reg on ALL anchors
(min-anchor 2025-04-01) with iters*1.1, saves opt_two_stage_full_clf/_reg.txt.
Then: python predict_test.py --spec "two_stage:opt_two_stage_full"
"""
from __future__ import annotations

import json
import os
from datetime import date

import lightgbm as lgb
import numpy as np

from common import MODELS_DIR, lean_load

DROP_COLS = ("user_id", "anchor_date", "target", "anchor_month", "anchor_doy")

meta = json.loads((MODELS_DIR / "opt_two_stage.meta.json").read_text())
bp = meta["params"]
clf_it = max(50, int(int(meta["clf_iter"]) * 1.1))
reg_it = max(50, int(int(meta["reg_iter"]) * 1.1))
print(f"refit opt_two_stage: clf {clf_it} it, reg {reg_it} it, params {bp}", flush=True)

full = lean_load("v2", min_anchor=date(2025, 4, 1), drop_cols=DROP_COLS)
X, y_raw, age = full["X_tr"], full["y_tr_raw"], full["age_days_tr"]
feat_cols = full["feat_cols"]
pos = y_raw > 0
w = None
if bp.get("use_hl"):
    w = np.power(0.5, age.astype(np.float64) / bp["half_life"])

common = dict(verbosity=-1, num_threads=int(os.environ.get("ML_THREADS", "48")), seed=42,
              bagging_freq=1, learning_rate=bp["learning_rate"],
              min_data_in_leaf=bp["min_data_in_leaf"], feature_fraction=bp["feature_fraction"],
              bagging_fraction=bp["bagging_fraction"], lambda_l2=bp["lambda_l2"])
clf = lgb.train(dict(objective="binary", num_leaves=bp["leaves_clf"], **common),
                lgb.Dataset(X, label=pos.astype(np.float32), weight=w, feature_name=feat_cols),
                num_boost_round=clf_it)
w_pos = w[pos] if w is not None else None
reg = lgb.train(dict(objective="regression", num_leaves=bp["leaves_reg"], **common),
                lgb.Dataset(X[pos], label=np.log1p(y_raw[pos]), weight=w_pos, feature_name=feat_cols),
                num_boost_round=reg_it)
clf.save_model(str(MODELS_DIR / "opt_two_stage_full_clf.txt"))
reg.save_model(str(MODELS_DIR / "opt_two_stage_full_reg.txt"))
print("saved opt_two_stage_full_clf/_reg", flush=True)
