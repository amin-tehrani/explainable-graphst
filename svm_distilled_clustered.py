# pip install scikit-learn joblib numpy
from dataclasses import dataclass
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from joblib import dump, load

@dataclass
class SVMDecoderConfig:
    C: float = 10.0                # SVM regularization
    gamma: str | float = "scale"   # "scale", "auto", or float
    class_weight: str | dict | None = "balanced"
    probability: bool = True       # enable Platt scaling for calibrated probs
    n_jobs: int = -1               # parallelize the K heads
    max_iter: int = -1             # -1 = no hard limit
    do_search: bool = False        # set True to run a small hyperparam search
    search_cv_folds: int = 3
    search_param_grid: dict | None = None

class SVMClusterDecoder(BaseEstimator, ClassifierMixin):
    """
    RBF-SVM One-vs-Rest decoder over embeddings -> cluster IDs.
    Builds K binary heads (one per class). Exposes predict and predict_proba.
    """
    def __init__(self, cfg: SVMDecoderConfig = SVMDecoderConfig()):
        self.cfg = cfg
        self.le_ = LabelEncoder()
        # pipeline: scale -> OvR(SVC_rbf)
        base = SVC(
            kernel="rbf",
            C=cfg.C,
            gamma=cfg.gamma,
            class_weight=cfg.class_weight,
            probability=cfg.probability,
            max_iter=cfg.max_iter,
        )
        self.pipeline_ = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", OneVsRestClassifier(base, n_jobs=cfg.n_jobs)),
        ])

    # scikit-learn API
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: [N, 64] float32/float64 embeddings
        y: [N] integer or string cluster IDs (K classes)
        """
        y_enc = self.le_.fit_transform(y)  # maps labels -> [0..K-1]
        if self.cfg.do_search:
            self._grid_search_and_set(X, y_enc)
        self.pipeline_.fit(X, y_enc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred_enc = self.pipeline_.predict(X)
        return self.le_.inverse_transform(y_pred_enc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns [N, K] calibrated per-class probabilities (OvR normalized)."""
        # For OvR, each binary SVM outputs p(class_k vs rest).
        # We renormalize to sum to 1 across K (simple normalization).
        proba_ovr = self.pipeline_.predict_proba(X)  # [N, K]
        s = proba_ovr.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return proba_ovr / s

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline_.decision_function(X)  # [N, K]

    def classes_(self):
        return self.le_.classes_

    def save(self, path: str):
        dump({"cfg": self.cfg, "le": self.le_, "pipe": self.pipeline_}, path)

    @staticmethod
    def load(path: str) -> "SVMClusterDecoder":
        obj = load(path)
        dec = SVMClusterDecoder(obj["cfg"])
        dec.le_ = obj["le"]
        dec.pipeline_ = obj["pipe"]
        return dec

    # ---- private helpers ----
    def _grid_search_and_set(self, X, y):
        # small, safe defaults for 64-D embeddings
        param_grid = self.cfg.search_param_grid or {
            "clf__estimator__C": [1.0, 10.0, 50.0],
            "clf__estimator__gamma": ["scale", 1/64, 1e-2],
        }
        cv = StratifiedKFold(n_splits=self.cfg.search_cv_folds, shuffle=True, random_state=42)
        gs = GridSearchCV(
            self.pipeline_,
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=self.cfg.n_jobs,
            refit=True,
            verbose=1,
        )
        gs.fit(X, y)
        self.pipeline_ = gs.best_estimator_
        # print(f"Best params: {gs.best_params_}, cv score: {gs.best_score_:.4f}")

# ------------------------ Usage ------------------------

# X_train: (N_train, 64) embeddings (float32/64)
# y_train: (N_train,) hard cluster labels from your slow clusterer
# X_val, y_val optional for quick check
def train_and_eval(X_train, y_train, X_val=None, y_val=None):
    cfg = SVMDecoderConfig(
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,        # enables calibrated probabilities
        n_jobs=-1,
        do_search=False,         # set True to grid-search C, gamma
    )
    dec = SVMClusterDecoder(cfg).fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        y_pred = dec.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"Val accuracy: {acc:.4f} on K={len(np.unique(y_train))} classes")

    return dec

# Example inference with uncertainty gating
def decode_with_reject(decoder: SVMClusterDecoder, X_new, p_thresh=0.6):
    probs = decoder.predict_proba(X_new)      # [N, K]
    pred_idx = probs.argmax(axis=1)
    pred = decoder.le_.inverse_transform(pred_idx)
    maxp = probs.max(axis=1)
    unsure = maxp < p_thresh                  # send these to the slow clusterer if desired
    return pred, probs, unsure


if __name__ == "__main__":
    from main import *

    base_encoder_path="graphst_base_encoder.pt"
    distilled_decoder_path="best_distilled_decoder.pt"
    device = torch.device("cpu")

    print("Base encoder path:", base_encoder_path)
    print("Distilled decoder path:", distilled_decoder_path)
    print("Device:", device)

    adata = load_dataset_V1_Human_Lymph_Node()
    annotate_graphclusters_V1_Human_Lymph_Node(adata)


    gst = get_graphst(adata, device)
    base_encoder = load_graphst_base_encoder(gst, base_encoder_path).to(device)
    
    h, _, _ ,_ ,_ ,_ = base_encoder(gst.features, gst.features_a, gst.adj)
    
    # Clustering

    # apply_clustering(graphst.gadata, method="leiden", n_clusters=7)
    clustering_key = "graphclust"

    clusters: pd.Series = gst.adata.obs[clustering_key]

    X_train, y_train = adata.X.toarray(), clusters.values
    res = train_and_eval(X_train, y_train)
    print(res)
    