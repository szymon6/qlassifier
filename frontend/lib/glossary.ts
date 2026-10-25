// Hover definitions for the dashboard's short labels (UAT, stage 1): native `title` + aria-label, no
// on-screen text (TDD §8.5). Model names come from the result; this file only expands them.
import type { Published } from "./types";

export const MODEL_TITLES: Record<string, string> = {
  qlassifier: "Qlassifier: product-state quantum kernel, self-tuned on exact features (ours)",
  classical_twin:
    "Classical twin: RBF kernel on sin/cos features, the same Gram matrix as geometry mode",
  zz_qsvc: "ZZ QSVC: QSVC on the Qiskit ZZ-feature-map fidelity kernel, the standard QML baseline",
  fqk_bandwidth_full: "FQK-bw full: bandwidth-tuned fidelity quantum kernel, full entanglement",
  fqk_bandwidth_linear:
    "FQK-bw linear: bandwidth-tuned fidelity quantum kernel, linear entanglement",
  pqk: "PQK: projected quantum kernel",
  separable: "Separable: product (separable) quantum kernel",
  logistic_regression: "Logistic regression (tuned)",
  rbf_svm: "RBF-SVM: support vector machine with a Gaussian (RBF) kernel (tuned)",
  random_forest: "Random forest (tuned)",
  gradient_boosting: "Histogram gradient boosting (tuned)",
  mlp: "MLP: multi-layer perceptron, a small neural network (tuned)",
  trivial: "Majority class: always predicts the most common class",
};

/** Full name of a model for a hover; the display name when none is known. */
export function modelTitle(id: string, name: string): string {
  return MODEL_TITLES[id] ?? name;
}

export const TERMS = {
  pp: "percentage points",
  ratio:
    "times fewer circuits than the bandwidth-tuned fidelity-kernel pipeline (training, prediction and its on-device search)",
  ci: "corrected 95% confidence interval over seeds (Nadeau–Bengio); ≈ best: tied with the top model",
  subset: "60/40: Part B, a paired Breast Cancer subset of 60 training and 40 test rows per seed",
  twoQ: "2Q gates: two-qubit gates per circuit after transpilation to the IBM device",
  circuits: "circuits to train and predict (hatched: on-device hyperparameter search), log scale",
  vsZz: "Qlassifier minus the standard QML baseline (ZZ QSVC), mean over seeds",
  noise: "largest mean accuracy loss over the three IBM device noise models (exact → device)",
  cost: "circuits of the bandwidth-tuned fidelity-kernel pipeline divided by Qlassifier's",
} as const;

export const METRIC_TITLES: Record<string, string> = {
  ba: "BA: balanced accuracy, the mean recall over the two classes (chance = 0.5)",
  acc: "accuracy: share of correct predictions",
  auc: "ROC-AUC: area under the ROC curve",
  f1: "F1 of the positive (minority) class",
};

export const RECIPE_TITLES: Record<string, string> = {
  pca8: "every model gets the same 8 PCA features, fitted on the training rows of each split",
  native: "every model gets the raw angle columns; geometry mode declares them as circles",
};

export function seedsTitle(n: number): string {
  return `${n} random train/test splits (seeds 0–${n - 1}), identical for every model`;
}

export function configsTitle(n: number): string {
  return `${n} hyperparameter configurations per tuned model, chosen by cross-validation inside the training split`;
}

export function publishedTitle(p: Published): string {
  const metric = p.metric === "balanced_accuracy" ? "BA" : p.metric;
  return `best published quantum result: ${metric} ${p.value.toFixed(3)}, ${p.method}, ${p.source} (other protocol, context only)`;
}
