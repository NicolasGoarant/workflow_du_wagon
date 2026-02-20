# frozen_string_literal: true
# Plot definitions for workflows — rendered by Chart.js
# Each plot has: type, title, data, options

WORKFLOW_PLOTS = {
  # ── EDA ──
  "eda_histogram" => {
    type: "bar", title: "Distribution des prix (histogramme)",
    data: {
      labels: ["0-100K","100-200K","200-300K","300-400K","400-500K","500-600K","600K+"],
      datasets: [{ label: "Nombre de maisons", data: [85, 420, 380, 265, 148, 95, 67],
                   backgroundColor: "rgba(34,197,94,0.6)", borderColor: "#22C55E", borderWidth: 1 }]
    },
    options: { annotation: "Distribution skewed à droite → envisager log(price)" }
  },
  "eda_boxplot_city" => {
    type: "boxplot_custom", title: "Prix par ville (boxplot simulé)",
    data: {
      labels: ["Paris", "Lyon", "Marseille", "Nancy"],
      datasets: [
        { label: "Médiane", data: [380000, 195000, 165000, 142000], backgroundColor: "rgba(34,197,94,0.8)" },
        { label: "Q1", data: [285000, 148000, 120000, 105000], backgroundColor: "rgba(34,197,94,0.3)" },
        { label: "Q3", data: [520000, 258000, 215000, 188000], backgroundColor: "rgba(34,197,94,0.3)" },
      ]
    },
    options: { annotation: "Paris nettement plus cher. Nancy le plus abordable." }
  },
  "eda_correlation" => {
    type: "heatmap_table", title: "Matrice de corrélation",
    data: {
      labels: ["surface", "rooms", "age", "price"],
      matrix: [
        [1.00, 0.72, -0.15, 0.81],
        [0.72, 1.00, -0.08, 0.68],
        [-0.15, -0.08, 1.00, -0.35],
        [0.81, 0.68, -0.35, 1.00]
      ]
    },
    options: { annotation: "surface-price: 0.81 (forte). surface-rooms: 0.72 (multicolinéarité à surveiller)" }
  },
  "eda_scatter" => {
    type: "scatter", title: "Surface vs Prix",
    data: {
      datasets: [{
        label: "Maisons",
        data: [
          {x:25,y:85},{x:35,y:95},{x:42,y:110},{x:48,y:125},{x:55,y:138},{x:60,y:155},
          {x:65,y:162},{x:70,y:178},{x:72,y:185},{x:78,y:195},{x:80,y:210},{x:85,y:220},
          {x:88,y:235},{x:90,y:245},{x:95,y:255},{x:98,y:262},{x:100,y:268},{x:105,y:280},
          {x:110,y:295},{x:115,y:310},{x:120,y:325},{x:125,y:340},{x:130,y:348},
          {x:140,y:375},{x:150,y:410},{x:160,y:445},{x:180,y:520},{x:200,y:580},
          {x:45,y:142},{x:55,y:120},{x:68,y:195},{x:75,y:168},{x:82,y:240},
          {x:92,y:230},{x:102,y:290},{x:112,y:275},{x:135,y:380},{x:155,y:390},
          {x:170,y:480},{x:190,y:540},{x:210,y:620},{x:250,y:780},{x:300,y:950},
        ],
        backgroundColor: "rgba(34,197,94,0.4)", borderColor: "#22C55E", pointRadius: 4,
      }]
    },
    options: { scales_x: "Surface (m²)", scales_y: "Prix (K€)", annotation: "Relation quasi-linéaire. Quelques outliers au-delà de 200m²." }
  },

  # ── LINEAR REGRESSION ──
  "linreg_actual_vs_pred" => {
    type: "scatter", title: "Actual vs Predicted",
    data: {
      datasets: [
        { label: "Prédictions", data: [
            {x:95,y:102},{x:142,y:138},{x:185,y:178},{x:210,y:218},{x:245,y:240},
            {x:268,y:272},{x:295,y:288},{x:320,y:335},{x:348,y:340},{x:380,y:368},
            {x:410,y:425},{x:455,y:438},{x:520,y:505},{x:580,y:562},{x:780,y:720},
          ],
          backgroundColor: "rgba(168,85,247,0.5)", borderColor: "#A855F7", pointRadius: 5 },
        { label: "Parfait (y=x)", data: [{x:80,y:80},{x:800,y:800}],
          type: "line", borderColor: "rgba(255,255,255,0.3)", borderDash: [5,5], pointRadius: 0 }
      ]
    },
    options: { scales_x: "Prix réel (K€)", scales_y: "Prix prédit (K€)", annotation: "Points proches de la diagonale → bonnes prédictions. R²=0.83" }
  },
  "linreg_coefficients" => {
    type: "bar_horizontal", title: "Coefficients (après scaling)",
    data: {
      labels: ["surface", "city_Paris", "rooms", "age", "garden", "city_Lyon"],
      datasets: [{ label: "Impact sur le prix (€)", data: [2850, 4500, 1200, -980, 650, 800],
                   backgroundColor: ["#22C55E","#22C55E","#22C55E","#EF4444","#22C55E","#22C55E"] }]
    },
    options: { annotation: "surface = feature la plus impactante. age = impact négatif." }
  },

  # ── LOGISTIC REGRESSION ──
  "logreg_roc" => {
    type: "line", title: "Courbe ROC (AUC = 0.94)",
    data: {
      labels: ["0.0","0.05","0.1","0.15","0.2","0.3","0.4","0.5","0.6","0.8","1.0"],
      datasets: [
        { label: "Modèle (AUC=0.94)", data: [0,0.52,0.68,0.76,0.82,0.88,0.92,0.94,0.96,0.98,1.0],
          borderColor: "#22C55E", backgroundColor: "rgba(34,197,94,0.1)", fill: true, tension: 0.3 },
        { label: "Hasard (0.50)", data: [0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0],
          borderColor: "rgba(255,255,255,0.3)", borderDash: [5,5], pointRadius: 0 }
      ]
    },
    options: { scales_x: "Taux de faux positifs", scales_y: "Taux de vrais positifs", annotation: "Plus la courbe est en haut à gauche, meilleur est le modèle." }
  },
  "logreg_confusion" => {
    type: "heatmap_table", title: "Matrice de confusion",
    data: {
      labels: ["Prédit: Pas spam", "Prédit: Spam"],
      row_labels: ["Réel: Pas spam", "Réel: Spam"],
      matrix: [[365, 17], [8, 43]]
    },
    options: { annotation: "FP=17 (mails bloqués à tort), FN=8 (spams passés). Recall spam = 84%." }
  },

  # ── TREES / RANDOM FOREST ──
  "trees_importance" => {
    type: "bar_horizontal", title: "Feature Importances (Random Forest)",
    data: {
      labels: ["surface", "rooms", "age", "city_Paris", "price_per_m2", "garden", "pool", "city_Lyon"],
      datasets: [{ label: "Importance", data: [0.342, 0.198, 0.156, 0.128, 0.089, 0.045, 0.028, 0.014],
                   backgroundColor: "rgba(34,197,94,0.6)", borderColor: "#22C55E", borderWidth: 1 }]
    },
    options: { annotation: "surface domine. ⚠️ Features corrélées se partagent l'importance." }
  },

  # ── BOOSTING ──
  "boosting_loss" => {
    type: "line", title: "Courbe d'entraînement XGBoost (early stopping)",
    data: {
      labels: (0..240).step(10).map(&:to_s),
      datasets: [
        { label: "Train logloss", data: [0.68,0.52,0.38,0.28,0.21,0.16,0.13,0.10,0.08,0.07,0.06,0.05,0.045,0.04,0.036,0.033,0.031,0.029,0.027,0.026,0.025,0.024,0.023,0.022,0.021],
          borderColor: "#A855F7", tension: 0.3 },
        { label: "Val logloss", data: [0.67,0.48,0.34,0.25,0.20,0.17,0.155,0.150,0.148,0.150,0.152,0.155,0.158,0.160,0.162,0.164,0.166,0.168,0.170,0.172,0.174,0.176,0.178,0.180,0.182],
          borderColor: "#22C55E", tension: 0.3 },
      ]
    },
    options: { scales_x: "Itérations", scales_y: "Log Loss", annotation: "Val_loss remonte après iter 80 → early stopping. Train continue à baisser → overfitting." }
  },

  # ── KNN ──
  "knn_k_curve" => {
    type: "line", title: "Choix de K — Train vs Test accuracy",
    data: {
      labels: (1..20).map(&:to_s),
      datasets: [
        { label: "Train", data: [1.0,0.96,0.94,0.93,0.92,0.91,0.905,0.90,0.895,0.89,0.885,0.88,0.875,0.87,0.865,0.86,0.855,0.85,0.845,0.84],
          borderColor: "#A855F7", tension: 0.3 },
        { label: "Test", data: [0.84,0.87,0.89,0.895,0.905,0.90,0.895,0.89,0.888,0.885,0.882,0.878,0.875,0.872,0.87,0.868,0.865,0.862,0.860,0.858],
          borderColor: "#22C55E", tension: 0.3 },
      ]
    },
    options: { scales_x: "K (nombre de voisins)", scales_y: "Accuracy", annotation: "K=5 : test accuracy max. K=1 : overfitting (train=100%). K>15 : underfitting." }
  },

  # ── K-MEANS ──
  "kmeans_elbow" => {
    type: "line", title: "Méthode du coude (Elbow)",
    data: {
      labels: (2..10).map(&:to_s),
      datasets: [
        { label: "Inertia", data: [12450, 8230, 5810, 4920, 4350, 3980, 3720, 3550, 3420],
          borderColor: "#22C55E", tension: 0.2, pointRadius: 6,
          pointBackgroundColor: ["#22C55E","#22C55E","#22C55E","#FBBF24","#22C55E","#22C55E","#22C55E","#22C55E","#22C55E"] },
      ]
    },
    options: { scales_x: "K (nombre de clusters)", scales_y: "Inertia", annotation: "Coude à K=4 — au-delà, le gain d'inertia est marginal." }
  },
  "kmeans_silhouette" => {
    type: "bar", title: "Silhouette Score par K",
    data: {
      labels: (2..10).map(&:to_s),
      datasets: [{ label: "Silhouette", data: [0.412, 0.485, 0.521, 0.498, 0.465, 0.440, 0.418, 0.395, 0.378],
                   backgroundColor: ["#22C55E","#22C55E","#FBBF24","#22C55E","#22C55E","#22C55E","#22C55E","#22C55E","#22C55E"] }]
    },
    options: { scales_x: "K", scales_y: "Score", annotation: "Max à K=4 → confirme l'elbow method." }
  },
  "kmeans_clusters" => {
    type: "scatter", title: "Projection 2D des clusters",
    data: {
      datasets: [
        { label: "Jeunes actifs", data: [{x:-2.1,y:1.3},{x:-1.8,y:0.9},{x:-2.4,y:1.1},{x:-1.5,y:1.5},{x:-2.0,y:0.7},{x:-1.7,y:1.4},{x:-2.2,y:1.0},{x:-1.9,y:1.2}],
          backgroundColor: "rgba(34,197,94,0.7)", pointRadius: 6 },
        { label: "Seniors fidèles", data: [{x:1.8,y:1.5},{x:2.1,y:1.2},{x:1.5,y:1.8},{x:2.3,y:1.0},{x:1.9,y:1.6},{x:2.0,y:1.3},{x:1.7,y:1.7}],
          backgroundColor: "rgba(168,85,247,0.7)", pointRadius: 6 },
        { label: "Gros budgets", data: [{x:0.5,y:-1.8},{x:0.8,y:-2.1},{x:0.3,y:-1.5},{x:0.9,y:-1.9},{x:0.6,y:-2.0},{x:0.7,y:-1.7}],
          backgroundColor: "rgba(251,191,36,0.7)", pointRadius: 6 },
        { label: "Inactifs", data: [{x:-0.5,y:-1.2},{x:-0.8,y:-0.9},{x:-0.3,y:-1.5},{x:-0.6,y:-1.0},{x:-0.4,y:-1.3}],
          backgroundColor: "rgba(239,68,68,0.7)", pointRadius: 6 },
      ]
    },
    options: { scales_x: "PC1", scales_y: "PC2", annotation: "4 groupes distincts. Les centroïdes sont bien séparés." }
  },

  # ── PCA ──
  "pca_variance" => {
    type: "bar", title: "Variance expliquée par composante",
    data: {
      labels: ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8"],
      datasets: [
        { label: "% variance", data: [38.2, 22.1, 15.4, 11.8, 8.1, 2.8, 1.1, 0.5],
          backgroundColor: "rgba(34,197,94,0.6)", borderColor: "#22C55E", borderWidth: 1 },
        { label: "% cumulé", data: [38.2, 60.3, 75.7, 87.5, 95.6, 98.4, 99.5, 100],
          type: "line", borderColor: "#FBBF24", tension: 0.3, pointRadius: 4 }
      ]
    },
    options: { scales_x: "Composantes", scales_y: "% Variance", annotation: "5 composantes captent 95.6% de l'info. On passe de 18 à 5 dimensions." }
  },

  # ── NEURAL NETWORKS ──
  "nn_training" => {
    type: "line", title: "Courbe d'entraînement (accuracy)",
    data: {
      labels: (1..25).map(&:to_s),
      datasets: [
        { label: "Train accuracy", data: [0.62,0.71,0.76,0.80,0.83,0.85,0.87,0.89,0.90,0.91,0.92,0.925,0.93,0.935,0.94,0.942,0.945,0.948,0.95,0.952,0.954,0.956,0.958,0.96,0.961],
          borderColor: "#A855F7", tension: 0.3 },
        { label: "Val accuracy", data: [0.58,0.68,0.74,0.78,0.81,0.84,0.86,0.88,0.895,0.905,0.91,0.915,0.918,0.920,0.922,0.923,0.924,0.923,0.922,0.921,0.920,0.919,0.918,0.917,0.916],
          borderColor: "#22C55E", tension: 0.3 },
      ]
    },
    options: { scales_x: "Epoch", scales_y: "Accuracy", annotation: "Val accuracy plafonne à epoch 17 (0.924). Early stopping restaure les poids de cette epoch." }
  },
  "nn_loss" => {
    type: "line", title: "Courbe de loss",
    data: {
      labels: (1..25).map(&:to_s),
      datasets: [
        { label: "Train loss", data: [0.65,0.48,0.38,0.31,0.26,0.22,0.19,0.17,0.15,0.14,0.13,0.12,0.11,0.105,0.10,0.095,0.09,0.088,0.085,0.082,0.08,0.078,0.076,0.074,0.072],
          borderColor: "#A855F7", tension: 0.3 },
        { label: "Val loss", data: [0.68,0.50,0.39,0.32,0.27,0.23,0.20,0.18,0.165,0.158,0.152,0.149,0.147,0.146,0.145,0.146,0.147,0.148,0.150,0.152,0.154,0.156,0.158,0.160,0.162],
          borderColor: "#22C55E", tension: 0.3 },
      ]
    },
    options: { scales_x: "Epoch", scales_y: "Loss", annotation: "Val loss remonte après epoch 15 → overfitting. EarlyStopping(patience=5) arrête à epoch 20." }
  },
}
