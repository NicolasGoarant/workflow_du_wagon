# frozen_string_literal: true

WORKFLOW_EXAMPLES = {
  "preprocessing" => {
    title: "🏠 Dataset immobilier — 5 appartements à nettoyer",
    dataframe: [
      ["surface", "pieces", "ville", "etage", "prix"],
      [65, 3, "Paris", 2, 285_000],
      [42, 2, "Lyon", nil, 158_000],
      [120, 5, "Paris", 4, nil],
      [65, 3, "Lyon", 0, 172_000],
      [42, 2, "Paris", 2, 285_000],
    ],
    problems: [
      "Ligne 2 : <code>etage</code> manquant (NaN) → imputer par la médiane (2.0)",
      "Ligne 3 : <code>prix</code> manquant → supprimer la ligne (c'est la target !)",
      "Lignes 1 et 5 : doublon exact → <code>drop_duplicates()</code>",
      "<code>ville</code> est catégoriel → One-Hot Encoding (ville_Paris, ville_Lyon)",
    ],
    result: [
      ["surface", "pieces", "etage", "ville_Lyon", "ville_Paris", "prix"],
      [65, 3, 2, 0, 1, 285_000],
      [42, 2, 2, 1, 0, 158_000],
      [65, 3, 0, 1, 0, 172_000],
    ],
    conclusion: "De 5 lignes × 5 colonnes à 3 lignes × 6 colonnes. Propre, numérique, prêt pour le modèle.",
  },

  "eda" => {
    title: "🏠 Exploration du dataset immobilier",
    dataframe: [
      ["surface", "pieces", "prix", "parking"],
      [35, 1, 125_000, 0],
      [65, 3, 285_000, 1],
      [42, 2, 158_000, 0],
      [110, 5, 520_000, 1],
      [78, 3, 310_000, 1],
    ],
    problems: [
      "<code>df.describe()</code> : surface de 35 à 110, prix de 125K à 520K — pas d'outlier flagrant",
      "<code>df.corr()</code> : corrélation surface↔prix = <strong>0.99</strong> (quasi parfaite, logique)",
      "Corrélation pieces↔surface = <strong>0.95</strong> → multicolinéarité, peut poser problème en régression",
      "Distribution de prix : skewed à droite → <code>np.log1p(prix)</code> pour normaliser",
    ],
    result: nil,
    conclusion: "L'EDA révèle que la surface est le meilleur prédicteur du prix, et qu'il y a un risque de multicolinéarité pieces/surface.",
  },

  "linreg" => {
    title: "🏠 Prédire le prix d'un appartement",
    dataframe: [
      ["surface", "pieces", "prix"],
      [35, 1, 125_000],
      [65, 3, 285_000],
      [42, 2, 158_000],
      [110, 5, 520_000],
    ],
    problems: [
      "Le modèle apprend : <code>prix = 4 890 × surface + 12 300 × pieces − 52 100</code>",
      "Coefficients : chaque m² supplémentaire → +4 890€, chaque pièce → +12 300€",
      "R² = 0.98 → le modèle explique 98% de la variance du prix",
    ],
    result: [
      ["surface", "pieces", "prix_réel", "prix_prédit"],
      [78, 3, "?", "~330 720€"],
    ],
    conclusion: "Le modèle prédit ~330K€ pour 78m², 3 pièces. RMSE = ~15 000€ → erreur moyenne de 15K€.",
  },

  "logreg" => {
    title: "📧 Détecter les spams dans une boîte mail",
    dataframe: [
      ["nb_liens", "mots_suspects", "longueur", "spam"],
      [0, 1, 450, 0],
      [8, 12, 120, 1],
      [1, 0, 800, 0],
      [15, 8, 95, 1],
      [2, 3, 350, 0],
    ],
    problems: [
      "Le modèle apprend : <code>P(spam) = σ(0.4 × nb_liens + 0.6 × mots_suspects − 0.01 × longueur − 2.1)</code>",
      "La sigmoïde σ transforme le score en probabilité [0, 1]",
      "Seuil par défaut = 0.5 : si P(spam) > 0.5 → classé spam",
    ],
    result: [
      ["nb_liens", "mots_suspects", "longueur", "P(spam)", "classé"],
      [5, 6, 200, "0.87", "🔴 Spam"],
    ],
    conclusion: "Nouvel email avec 5 liens et 6 mots suspects → P(spam) = 87% → classé spam. On pourrait baisser le seuil à 0.3 pour ne rien rater.",
  },

  "trees" => {
    title: "🎓 Prédire si un étudiant réussit l'examen",
    dataframe: [
      ["heures_étude", "cours_suivis", "exercices_faits", "réussite"],
      [2, 3, 5, 0],
      [8, 10, 20, 1],
      [5, 7, 12, 1],
      [1, 2, 3, 0],
      [6, 8, 15, 1],
    ],
    problems: [
      "L'arbre apprend des règles : <strong>SI heures_étude > 4 ET exercices > 10 → réussite</strong>",
      "Pas besoin de scaling (seuls les seuils comptent)",
      "Random Forest : 100 arbres votent → plus robuste qu'un seul arbre",
      "<code>feature_importances_</code> : exercices (0.52) > heures (0.31) > cours (0.17)",
    ],
    result: [
      ["heures_étude", "cours_suivis", "exercices_faits", "prédiction"],
      [4, 6, 11, "✅ Réussit (78 arbres sur 100 votent oui)"],
    ],
    conclusion: "Le nombre d'exercices faits est le facteur n°1. L'arbre le montre clairement — plus interprétable qu'un réseau de neurones.",
  },

  "boosting" => {
    title: "🏠 Prix immobilier — pousser la performance",
    dataframe: [
      ["surface", "pieces", "etage", "parking", "prix"],
      [35, 1, 5, 0, 125_000],
      [65, 3, 2, 1, 285_000],
      [42, 2, 0, 0, 158_000],
      [110, 5, 4, 1, 520_000],
    ],
    problems: [
      "Arbre 1 prédit tout à 272K (moyenne). Erreurs : −147K, +13K, −114K, +248K",
      "Arbre 2 se concentre sur les grosses erreurs → corrige de lr × erreur",
      "Avec lr=0.1 et 500 arbres → chaque arbre fait un petit pas correctif",
      "Early stopping : arrête à l'arbre 342 (la val_loss remontait)",
    ],
    result: [
      ["surface", "pieces", "etage", "parking", "prix_prédit"],
      [78, 3, 3, 1, "~318 500€"],
    ],
    conclusion: "XGBoost : RMSE = 8 200€ vs 15 000€ pour la régression linéaire. Le boosting gagne 45% d'erreur en moins.",
  },

  "knn" => {
    title: "🍷 Classifier un vin (rouge / blanc / rosé)",
    dataframe: [
      ["acidité", "sucre", "alcool", "type"],
      [7.4, 1.9, 11.5, "rouge"],
      [6.8, 5.2, 10.0, "blanc"],
      [7.1, 4.8, 9.5, "blanc"],
      [7.5, 1.5, 12.0, "rouge"],
      [6.9, 3.8, 11.0, "rosé"],
    ],
    problems: [
      "⚠️ Scaling obligatoire : acidité [6.8–7.5] vs sucre [1.5–5.2] → StandardScaler",
      "K=3 : pour un nouveau vin, on mesure la distance avec TOUS les vins connus",
      "On prend les 3 plus proches et on vote la majorité",
      "Le 'modèle' = le dataset entier stocké en mémoire (lazy learner)",
    ],
    result: [
      ["acidité", "sucre", "alcool", "3 voisins", "prédiction"],
      [7.0, 4.0, 10.5, "blanc, blanc, rosé", "🍷 Blanc (2 votes sur 3)"],
    ],
    conclusion: "Simple et efficace. Mais avec 1M de vins, chaque prédiction recalcule 1M de distances → lent.",
  },

  "svm" => {
    title: "🏥 Diagnostic tumeur (bénigne / maligne)",
    dataframe: [
      ["taille_noyau", "texture", "périmètre", "diagnostic"],
      [13.5, 14.2, 87, "bénigne"],
      [20.1, 23.5, 132, "maligne"],
      [12.4, 15.7, 82, "bénigne"],
      [18.2, 21.0, 120, "maligne"],
    ],
    problems: [
      "⚠️ Scaling obligatoire (basé sur des distances)",
      "SVM cherche la frontière (hyperplan) qui <strong>maximise la marge</strong> entre les classes",
      "Kernel RBF : projette les données dans un espace supérieur si pas linéairement séparables",
      "Ne retient que les <strong>support vectors</strong> (points proches de la frontière)",
    ],
    result: [
      ["taille_noyau", "texture", "périmètre", "prédiction"],
      [16.0, 19.0, 105, "⚠️ Maligne (marge étroite → incertitude)"],
    ],
    conclusion: "SVM excelle sur les petits datasets haute dimension. Ici 3 features, 4 lignes — il trouve la frontière.",
  },

  "kmeans" => {
    title: "🛒 Segmenter les clients d'un e-commerce",
    dataframe: [
      ["âge", "dépense_mois", "fréquence_achat"],
      [22, 45, 12],
      [55, 320, 3],
      [28, 60, 15],
      [48, 280, 4],
      [25, 50, 10],
    ],
    problems: [
      "⚠️ Scaling obligatoire (âge [22–55] vs dépense [45–320])",
      "K-Means avec K=2 (choisi par méthode du coude)",
      "Itère : place 2 centroïdes → assigne → recalcule → jusqu'à convergence",
      "Pas de labels — c'est au Data Scientist d'interpréter les clusters",
    ],
    result: [
      ["âge", "dépense", "fréquence", "cluster", "interprétation"],
      ["22, 28, 25", "45–60€", "10–15×", "Cluster 0", "🛍️ Jeunes actifs, petits achats fréquents"],
      ["55, 48", "280–320€", "3–4×", "Cluster 1", "💎 Seniors, gros achats rares"],
    ],
    conclusion: "Deux profils clients identifiés sans labels. Le marketing peut adapter ses campagnes par segment.",
  },

  "pca" => {
    title: "📊 Compresser 5 features en 2 composantes",
    dataframe: [
      ["surface", "pieces", "salles_bain", "balcon_m2", "etage"],
      [65, 3, 1, 5, 2],
      [120, 5, 2, 12, 4],
      [42, 2, 1, 3, 1],
      [95, 4, 2, 8, 3],
    ],
    problems: [
      "⚠️ Scaling obligatoire avant PCA",
      "5 features corrélées → PCA trouve 2 axes qui captent 94% de la variance",
      "PC1 = 0.52×surface + 0.48×pieces + 0.45×salles_bain + ... → axe 'taille globale'",
      "PC2 = 0.70×etage − 0.30×balcon + ... → axe 'hauteur vs extérieur'",
    ],
    result: [
      ["PC1 (taille)", "PC2 (hauteur)", "variance captée"],
      ["-0.82", "0.31", "PC1 : 78%"],
      ["1.95", "0.67", "PC2 : 16%"],
      ["-1.54", "-0.43", "Total : 94%"],
      ["0.41", "-0.55", ""],
    ],
    conclusion: "De 5 dimensions à 2, en ne perdant que 6% d'info. On peut maintenant visualiser les données en 2D.",
  },

  "nn" => {
    title: "🎵 Prédire le genre musical d'un morceau",
    dataframe: [
      ["tempo", "énergie", "dansabilité", "acoustique", "genre"],
      [120, 0.85, 0.72, 0.10, "pop"],
      [140, 0.95, 0.60, 0.05, "rock"],
      [90, 0.30, 0.45, 0.85, "classique"],
      [128, 0.78, 0.88, 0.15, "électro"],
    ],
    problems: [
      "Architecture : Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dense(4, softmax)",
      "4 inputs → 64 neurones → 32 neurones → 4 classes = <strong>2 340 paramètres</strong>",
      "Loss : <code>categorical_crossentropy</code> (multi-classes)",
      "Entraîné 50 epochs, early stopping à l'epoch 38",
    ],
    result: [
      ["tempo", "énergie", "dansabilité", "acoustique", "P(pop)", "P(rock)", "P(classique)", "P(électro)"],
      [125, 0.80, 0.75, 0.12, "0.62", "0.18", "0.03", "0.17"],
    ],
    conclusion: "Le réseau sort des probabilités par classe. Le morceau est classé 'pop' avec 62% de confiance.",
  },

  "cnn" => {
    title: "🐱 Classifier des images : chat ou chien ?",
    dataframe: [
      ["image", "taille", "canaux", "label"],
      ["chat_01.jpg", "224×224", "RGB (3)", "chat"],
      ["chien_01.jpg", "224×224", "RGB (3)", "chien"],
      ["chat_02.jpg", "224×224", "RGB (3)", "chat"],
      ["chien_02.jpg", "224×224", "RGB (3)", "chien"],
    ],
    problems: [
      "Input : 224×224×3 = <strong>150 528 valeurs</strong> par image (pixels RGB normalisés /255)",
      "Transfer Learning : VGG16 pré-entraîné (base.trainable = False)",
      "On ajoute : Flatten → Dense(128, relu) → Dropout(0.5) → Dense(1, sigmoid)",
      "Le CNN apprend : bords → textures → formes → oreilles pointues = chat",
    ],
    result: [
      ["image", "P(chat)", "P(chien)", "prédiction"],
      ["test_01.jpg", "0.92", "0.08", "🐱 Chat"],
    ],
    conclusion: "Avec seulement 100 images d'entraînement, le transfer learning atteint 94% d'accuracy grâce aux features pré-apprises.",
  },

  "rnn" => {
    title: "📈 Prédire la température de demain",
    dataframe: [
      ["jour", "temp", "humidité", "vent"],
      ["Lun", 12.5, 65, 15],
      ["Mar", 13.0, 60, 12],
      ["Mer", 14.2, 55, 10],
      ["Jeu", 13.8, 58, 14],
      ["Ven", "?", "?", "?"],
    ],
    problems: [
      "Séquence de 4 jours → prédire le 5e. Le LSTM retient le contexte temporel",
      "Input shape : (batch, 4 timesteps, 3 features)",
      "LSTM(64) → les portes forget/input/output gèrent la mémoire long terme",
      "Un RNN classique oublierait le lundi ; le LSTM le retient si c'est pertinent",
    ],
    result: [
      ["jour", "temp_prédite"],
      ["Ven", "~14.0°C"],
    ],
    conclusion: "Le LSTM capte la tendance ascendante (12.5 → 14.2) et prédit ~14.0°C. Plus la séquence est longue, plus le contexte est riche.",
  },

  "pipeline" => {
    title: "🔧 Tout assembler dans un Pipeline",
    dataframe: [
      ["surface", "ville", "etage", "prix"],
      [65, "Paris", 2, 285_000],
      [42, "Lyon", 0, 158_000],
      [110, "Paris", 4, 520_000],
    ],
    problems: [
      "Numérique (surface, etage) : <code>SimpleImputer(median)</code> → <code>StandardScaler()</code>",
      "Catégoriel (ville) : <code>SimpleImputer(most_frequent)</code> → <code>OneHotEncoder()</code>",
      "<code>ColumnTransformer</code> applique le bon preprocessing à chaque type",
      "<code>Pipeline([('preproc', ct), ('model', Ridge())])</code> → un seul objet",
    ],
    result: [
      ["Code", ""],
      ["pipe.fit(X_train, y_train)", "Fit le scaler + le modèle en 1 ligne"],
      ["pipe.predict(X_new)", "Transforme + prédit automatiquement"],
      ["joblib.dump(pipe, 'pipe.pkl')", "Sauvegarde tout (preprocessing + modèle)"],
    ],
    conclusion: "Un Pipeline garantit que le preprocessing est identique en train et en production. Zéro risque de data leakage.",
  },

  "mlops" => {
    title: "🚀 Du notebook au serveur de production",
    dataframe: [
      ["Étape", "Outil", "Ce que ça fait"],
      ["1. Entraîner", "MLflow", "Log les params, métriques, et le modèle"],
      ["2. Versionner", "MLflow Registry", "Tague le meilleur modèle 'Production'"],
      ["3. Packager", "Docker", "Crée un conteneur avec toutes les dépendances"],
      ["4. Déployer", "API (FastAPI)", "Expose /predict en endpoint HTTP"],
      ["5. Automatiser", "Prefect", "Ré-entraîne automatiquement chaque semaine"],
    ],
    problems: [
      "<code>mlflow.log_param('lr', 0.1)</code> → traçabilité des expériences",
      "<code>mlflow.sklearn.log_model(pipe, 'model')</code> → modèle versionné",
      "Docker : <code>FROM python:3.10</code> → même environnement partout",
      "Prefect : <code>@flow</code> + <code>@task</code> → orchestration automatisée",
    ],
    result: nil,
    conclusion: "Le modèle passe de ton notebook Jupyter à une API que n'importe qui peut appeler avec un simple curl.",
  },
}
