# frozen_string_literal: true

ML_METHODS = [
  # ── PHASE 1 : APPRENDRE ──
  {
    method: ".fit(X_train, y_train)",
    phase: "Apprendre",
    phase_icon: "🧠",
    phase_color: "purple",
    short: "Apprend des données. Mémorise les paramètres internes.",
    explain: "C'est LE moment où le modèle regarde les données et en tire quelque chose. Un scaler mémorise la moyenne et l'écart-type. Un arbre mémorise les règles de décision. Un réseau de neurones ajuste ses poids.",
    examples: [
      { context: "Scaler", code: "scaler.fit(X_train)", learns: "Mémorise μ=102 et σ=48 pour la colonne surface" },
      { context: "Encoder", code: 'encoder.fit(X_train[["city"]])', learns: "Mémorise les catégories : Paris, Lyon, Marseille, Nancy" },
      { context: "Modèle", code: "model.fit(X_train, y_train)", learns: "Apprend les coefficients / poids / règles" },
      { context: "PCA", code: "pca.fit(X_train)", learns: "Trouve les axes de variance maximale" },
      { context: "KMeans", code: "km.fit(X_train)", learns: "Place les centroïdes des clusters" },
    ],
    rule: "⚠️ TOUJOURS sur X_train (jamais X_test). Sinon = data leakage.",
    frameworks: ["sklearn", "keras", "xgboost"],
  },
  {
    method: ".compile(optimizer, loss, metrics)",
    phase: "Apprendre",
    phase_icon: "🧠",
    phase_color: "purple",
    short: "Configure le mode d'apprentissage AVANT le .fit(). Keras uniquement.",
    explain: "Le .compile() ne touche pas aux données. Il dit au réseau : 'Voilà comment tu vas apprendre'. C'est le mode d'emploi du .fit(). On choisit l'optimizer (comment ajuster les poids), la loss (quoi minimiser), et les métriques (quoi surveiller).",
    examples: [
      { context: "Classification binaire", code: 'model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])', learns: "Adam + crossentropy + accuracy" },
      { context: "Régression", code: 'model.compile(optimizer="adam", loss="mse", metrics=["mae"])', learns: "Adam + MSE + MAE" },
      { context: "Multi-classes", code: 'model.compile(optimizer="adam", loss="categorical_crossentropy")', learns: "Adam + categorical crossentropy" },
    ],
    rule: "Obligatoire AVANT .fit() en Keras. N'existe pas en sklearn (tout est dans le constructeur).",
    frameworks: ["keras"],
  },

  # ── PHASE 2 : TRANSFORMER ──
  {
    method: ".transform(X)",
    phase: "Transformer",
    phase_icon: "🔄",
    phase_color: "cyan",
    short: "Applique la transformation apprise au .fit(). Ne ré-apprend PAS.",
    explain: "L'objet utilise ce qu'il a mémorisé au .fit() pour transformer de nouvelles données. Le scaler applique (x − μ) / σ avec le μ et σ du train. L'encoder remplace les catégories par les codes appris.",
    examples: [
      { context: "Scaler", code: "X_test_sc = scaler.transform(X_test)", learns: "Applique (x − 102) / 48 (valeurs du train)" },
      { context: "Encoder", code: "encoded = encoder.transform(X_test)", learns: "Encode avec les catégories vues au fit" },
      { context: "PCA", code: "X_pca = pca.transform(X_test)", learns: "Projette sur les axes trouvés au fit" },
    ],
    rule: "⚠️ Ne JAMAIS utiliser .fit_transform() sur X_test. Toujours .transform() seul.",
    frameworks: ["sklearn"],
  },
  {
    method: ".fit_transform(X_train)",
    phase: "Transformer",
    phase_icon: "🔄",
    phase_color: "cyan",
    short: "Raccourci : .fit() + .transform() en une ligne. UNIQUEMENT sur le train.",
    explain: "Fait les deux opérations d'un coup : apprend les paramètres ET applique la transformation. C'est un raccourci pratique, mais il ne doit être utilisé QUE sur le train. Sur le test, on utilise .transform() seul.",
    examples: [
      { context: "Train", code: "X_train_sc = scaler.fit_transform(X_train)", learns: "Apprend μ/σ du train ET transforme" },
      { context: "Test", code: "X_test_sc = scaler.transform(X_test)", learns: "Réutilise μ/σ du train (pas de fit !)" },
    ],
    rule: "⚠️ RÉSERVÉ au train set. C'est la source #1 de data leakage chez les débutants.",
    frameworks: ["sklearn"],
  },

  # ── PHASE 3 : PRÉDIRE ──
  {
    method: ".predict(X)",
    phase: "Prédire",
    phase_icon: "🎯",
    phase_color: "green",
    short: "Donne la réponse du modèle : une classe ou une valeur.",
    explain: "Le modèle utilise ce qu'il a appris pour donner une réponse. En classification, c'est la classe (0 ou 1). En régression, c'est la valeur (245 000 €). En Keras, c'est la probabilité brute (il faut arrondir soi-même).",
    examples: [
      { context: "Régression", code: "y_pred = model.predict(X_test)", learns: "[245000, 182000, 320000, ...]" },
      { context: "Classification sklearn", code: "y_pred = model.predict(X_test)", learns: "[0, 1, 0, 1, ...] (classes directement)" },
      { context: "Classification Keras", code: "probas = model.predict(X_test)", learns: "[0.92, 0.15, 0.87, ...] (probabilités)" },
      { context: "KMeans", code: "labels = km.predict(X_new)", learns: "[2, 0, 3, 1, ...] (numéro de cluster)" },
    ],
    rule: "Toujours après .fit(). Si le modèle n'a pas appris, erreur.",
    frameworks: ["sklearn", "keras", "xgboost"],
  },
  {
    method: ".predict_proba(X)",
    phase: "Prédire",
    phase_icon: "🎯",
    phase_color: "green",
    short: "Donne la probabilité de chaque classe. Sklearn uniquement.",
    explain: "Au lieu de dire '1' ou '0', donne la confiance du modèle. Indispensable pour ajuster le seuil de décision, tracer la courbe ROC, et évaluer l'AUC.",
    examples: [
      { context: "Binaire", code: "probas = model.predict_proba(X_test)[:, 1]", learns: "[0.92, 0.15, 0.87, 0.34, ...]" },
      { context: "Seuil custom", code: "y_pred = (probas >= 0.3).astype(int)", learns: "Seuil abaissé → plus de détection, moins de precision" },
    ],
    rule: "Le [:, 1] prend la proba de la classe positive. Indispensable pour ROC/AUC.",
    frameworks: ["sklearn"],
  },
  {
    method: ".fit_predict(X)",
    phase: "Prédire",
    phase_icon: "🎯",
    phase_color: "green",
    short: "Raccourci : .fit() + .predict() en une ligne. Pour le clustering.",
    explain: "Spécifique au non-supervisé (KMeans, DBSCAN). Apprend les clusters ET assigne les labels en une fois.",
    examples: [
      { context: "KMeans", code: "labels = km.fit_predict(X_sc)", learns: "[0, 2, 1, 3, 0, 2, ...]" },
    ],
    rule: "Équivalent de km.fit(X) puis km.predict(X), mais plus concis.",
    frameworks: ["sklearn"],
  },

  # ── PHASE 4 : ÉVALUER ──
  {
    method: ".score(X, y)",
    phase: "Évaluer",
    phase_icon: "📏",
    phase_color: "yellow",
    short: "Évalue la performance. Accuracy (classif) ou R² (régression). Sklearn uniquement.",
    explain: "Raccourci qui fait predict + calcul de la métrique par défaut. Pour la classification c'est l'accuracy, pour la régression c'est le R². Pratique mais limité — souvent on préfère les métriques spécifiques.",
    examples: [
      { context: "Train", code: "model.score(X_train, y_train)", learns: "0.952 (accuracy ou R² sur le train)" },
      { context: "Test", code: "model.score(X_test, y_test)", learns: "0.918 (performance réelle)" },
      { context: "Comparaison", code: "train vs test proche → pas d'overfitting", learns: "" },
    ],
    rule: "Train >> Test = overfitting. Train ≈ Test = modèle stable.",
    frameworks: ["sklearn"],
  },
  {
    method: ".evaluate(X, y)",
    phase: "Évaluer",
    phase_icon: "📏",
    phase_color: "yellow",
    short: "Calcule la loss ET les métriques sur un dataset. Keras uniquement.",
    explain: "Équivalent Keras du .score() mais plus riche : renvoie la loss (ce que le modèle minimise) ET toutes les métriques définies au .compile().",
    examples: [
      { context: "Évaluation", code: "loss, acc = model.evaluate(X_test, y_test)", learns: "loss=0.147, accuracy=0.924" },
    ],
    rule: "Toujours sur le test set pour la performance finale.",
    frameworks: ["keras"],
  },
  {
    method: "cross_val_score(model, X, y, cv=5)",
    phase: "Évaluer",
    phase_icon: "📏",
    phase_color: "yellow",
    short: "Évalue K fois en changeant le fold de validation. Mesure la stabilité.",
    explain: "Découpe les données en K morceaux. Entraîne K fois en laissant 1 morceau de côté. Renvoie K scores → la moyenne et l'écart-type mesurent la fiabilité du modèle.",
    examples: [
      { context: "Cross-val", code: 'cv = cross_val_score(model, X, y, cv=5, scoring="r2")', learns: "[0.82, 0.85, 0.83, 0.84, 0.81]" },
      { context: "Résumé", code: 'f"R² = {cv.mean():.3f} ± {cv.std():.3f}"', learns: "R² = 0.830 ± 0.014 → stable" },
    ],
    rule: "std > 0.05 → modèle instable. std < 0.02 → très fiable.",
    frameworks: ["sklearn"],
  },

  # ── UTILITAIRES ──
  {
    method: ".get_params() / .set_params()",
    phase: "Utilitaire",
    phase_icon: "🔧",
    phase_color: "gray",
    short: "Lire ou modifier les hyperparamètres du modèle.",
    explain: "Utile pour inspecter un modèle ou dans un pipeline avec GridSearchCV (notation double underscore).",
    examples: [
      { context: "Lire", code: "model.get_params()", learns: "{n_estimators: 100, max_depth: 10, ...}" },
      { context: "Modifier", code: "model.set_params(max_depth=5)", learns: "Change sans recréer l'objet" },
    ],
    rule: "set_params ne ré-entraîne pas. Il faut refaire .fit() après.",
    frameworks: ["sklearn"],
  },
  {
    method: ".summary()",
    phase: "Utilitaire",
    phase_icon: "🔧",
    phase_color: "gray",
    short: "Affiche l'architecture du réseau (couches, paramètres). Keras uniquement.",
    explain: "Vue d'ensemble du réseau : chaque couche, sa forme de sortie, et le nombre de paramètres entraînables.",
    examples: [
      { context: "Dense", code: "model.summary()", learns: "Dense(64) → 1216 params, Dropout → 0, Dense(1) → 65" },
    ],
    rule: "Vérifier le nombre total de paramètres. Trop → overfitting. Pas assez → underfitting.",
    frameworks: ["keras"],
  },
]

# Also add to GLOSSARY
ML_METHODS.each do |m|
  GLOSSARY.push({
    term: m[:method],
    category: "Méthodes",
    definition: m[:short],
    code: m[:examples]&.first&.dig(:code),
    workflow: case m[:phase]
             when "Apprendre" then "preprocessing"
             when "Transformer" then "preprocessing"
             when "Prédire" then "linreg"
             when "Évaluer" then "linreg"
             else "pipeline"
             end
  })
end
