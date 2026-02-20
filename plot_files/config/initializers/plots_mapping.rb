# frozen_string_literal: true
# Maps plot_id to [workflow_slug, step_index]
# step_index is 0-based

STEP_PLOTS = {
  # EDA
  ["eda", 1] => ["eda_histogram"],                    # Distributions
  ["eda", 2] => ["eda_scatter", "eda_boxplot_city"],   # Relations bivariées
  ["eda", 3] => ["eda_correlation"],                   # Matrice de corrélation

  # Linear Regression
  ["linreg", 1] => ["linreg_coefficients"],            # Coefficients
  ["linreg", 3] => ["linreg_actual_vs_pred"],          # Évaluer

  # Logistic Regression
  ["logreg", 1] => ["logreg_confusion"],               # Entraîner et évaluer
  ["logreg", 2] => ["logreg_roc"],                     # ROC curve

  # Trees
  ["trees", 0] => ["trees_importance"],                # Feature importance

  # Boosting
  ["boosting", 0] => ["boosting_loss"],                # Early stopping

  # KNN
  ["knn", 0] => ["knn_k_curve"],                       # Choix de K

  # K-Means
  ["kmeans", 0] => ["kmeans_elbow", "kmeans_silhouette"],  # Choisir K
  ["kmeans", 1] => ["kmeans_clusters"],                     # Analyser clusters

  # PCA
  ["pca", 0] => ["pca_variance"],                      # Variance expliquée

  # Neural Networks
  ["nn", 1] => ["nn_training", "nn_loss"],             # Entraîner
}
