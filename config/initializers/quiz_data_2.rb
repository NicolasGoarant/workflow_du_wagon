# frozen_string_literal: true
# Quiz questions 41-103 — positions randomisées, longueurs équilibrées

QUIZ_QUESTIONS.push(

  # ── PREPROCESSING (12 questions supplémentaires) ──

  { id: 41, category: "Preprocessing", difficulty: "facile",
    question: "Que fait <code>df.isna().sum()</code> ?",
    choices: ["Supprime les NaN, selon la documentation sklearn", "Remplace les NaN par 0, indépendamment du type de modèle utilisé", "Compte les valeurs manquantes par colonne", "Compte les lignes du DataFrame"],
    answer: 2,
    explanation: "isna() crée un masque True/False pour chaque cellule, puis sum() compte les True par colonne. Premier réflexe pour diagnostiquer les données manquantes." },

  { id: 42, category: "Preprocessing", difficulty: "moyen",
    question: "Pourquoi séparer <code>train_test_split</code> AVANT le preprocessing ?",
    choices: ["Parce que sklearn l'exige, pour tous les modèles", "Pour avoir des sets de taille égale", "Pour gagner du temps de calcul, dans la majorité des pipelines", "Pour éviter le data leakage"],
    answer: 3,
    explanation: "Si tu normalises AVANT le split, le scaler voit les données de test → le modèle a indirectement 'vu' le test. C'est du data leakage. Toujours : split → fit(train) → transform(test)." },

  { id: 43, category: "Preprocessing", difficulty: "difficile",
    question: "Quelle est la différence entre <code>StandardScaler</code> et <code>MinMaxScaler</code> ?",
    choices: ["MinMaxScaler est plus rapide", "StandardScaler ne fonctionne qu'avec des entiers", "Aucune différence, c'est une idée reçue mais qui ne s'applique pas ici", "StandardScaler centre à μ=0, σ=1"],
    answer: 3,
    explanation: "StandardScaler : (x − μ) / σ → distribution centrée réduite. MinMaxScaler : (x − min) / (max − min) → entre 0 et 1. StandardScaler est préféré quand il y a des outliers (moins sensible aux extrêmes)." },

  { id: 44, category: "Preprocessing", difficulty: "moyen",
    question: "Tu as une colonne <code>taille</code> avec les valeurs S, M, L, XL. Quel encodage utiliser ?",
    choices: ["Supprimer la colonne, selon les bonnes pratiques", "One-Hot Encoding, dans la majorité des pipelines", "Label Encoding aléatoire, sans tenir compte des spécificités du problème", "Ordinal Encoding car il y a un ordre naturel"],
    answer: 3,
    explanation: "S < M < L < XL → il y a un ordre. Ordinal Encoding (S=1, M=2, L=3, XL=4) préserve cette hiérarchie. One-Hot perdrait l'information d'ordre." },

  { id: 45, category: "Preprocessing", difficulty: "facile",
    question: "Que fait <code>df.drop_duplicates()</code> ?",
    choices: ["Supprime les outliers, mais ce n'est pas la meilleure approche pour ce cas", "Supprime les colonnes en double", "Supprime les valeurs NaN, cette méthode est utilisée dans les pipelines classiques", "Supprime les lignes entièrement identiques"],
    answer: 3,
    explanation: "Retire les lignes où TOUTES les valeurs sont identiques à une autre ligne. Avec subset=['col1', 'col2'], on peut ne vérifier que certaines colonnes." },

  { id: 46, category: "Preprocessing", difficulty: "difficile",
    question: "Quelle méthode détecte les outliers avec l'IQR ?",
    choices: ["Supprimer les valeurs > moyenne", "Valeurs hors de [Q1 − 1.5×IQR, Q3 + 1.5×IQR]", "Toute valeur > 2 écarts-types", "Les valeurs NaN sont des outliers"],
    answer: 1,
    explanation: "IQR = Q3 − Q1. Les valeurs en dessous de Q1 − 1.5×IQR ou au-dessus de Q3 + 1.5×IQR sont considérées comme outliers. C'est la méthode des boxplots." },

  { id: 47, category: "Preprocessing", difficulty: "moyen",
    question: "Pourquoi imputer la target (y) manquante est-il une mauvaise idée ?",
    choices: ["C'est une bonne idée en fait", "La target n'a jamais de NaN", "C'est trop lent, en règle générale", "On inventerait des réponses fausses"],
    answer: 3,
    explanation: "Imputer la target = inventer la réponse. Le modèle apprendrait sur des données fictives. Toujours supprimer les lignes où la target est manquante." },

  { id: 48, category: "Preprocessing", difficulty: "facile",
    question: "Que renvoie <code>df.dtypes</code> ?",
    choices: ["Les corrélations, c'est la méthode standard", "Le nombre de valeurs uniques", "Le type de chaque colonne", "Les statistiques descriptives"],
    answer: 2,
    explanation: "dtypes montre le type de chaque colonne. 'object' = souvent du texte (catégoriel), 'int64'/'float64' = numérique. Essentiel pour savoir quoi encoder." },

  { id: 49, category: "Preprocessing", difficulty: "moyen",
    question: "Tu as 200 catégories dans une colonne 'ville'. One-Hot Encoding est-il une bonne idée ?",
    choices: ["Oui, toujours, dans la plupart des cas", "Non — 200 colonnes binaires = explosion dimensionnelle", "Oui mais seulement en régression, comme le recommande la documentation officielle de sklearn", "Ça dépend du processeur, pour tous les modèles"],
    answer: 1,
    explanation: "One-Hot sur 200 villes crée 200 colonnes. La matrice devient très sparse, le modèle ralentit et peut overfitter. Alternatives : Target Encoding, regrouper les villes rares, ou embeddings." },

  { id: 50, category: "Preprocessing", difficulty: "difficile",
    question: "Que fait <code>SimpleImputer(strategy='most_frequent')</code> ?",
    choices: ["Remplace les NaN par la valeur la plus fréquente", "Remplace les NaN par 0, dans la plupart des cas", "Remplace les NaN par la moyenne, comme le recommande la documentation officielle de sklearn", "Supprime les lignes avec NaN, mais ce n'est pas la meilleure approche pour ce cas"],
    answer: 0,
    explanation: "Le mode (valeur la plus fréquente) est la stratégie par défaut pour les colonnes catégorielles. Pour les numériques, on préfère 'median' (robuste aux outliers) ou 'mean'." },

  { id: 51, category: "Preprocessing", difficulty: "moyen",
    question: "À quoi sert <code>ColumnTransformer</code> ?",
    choices: ["Appliquer des transformations différentes selon le type de colonne", "Supprimer les colonnes inutiles, selon la documentation sklearn", "Renommer les colonnes, dans la plupart des cas", "Transformer toutes les colonnes de la même façon"],
    answer: 0,
    explanation: "ColumnTransformer applique un pipeline numérique (imputer + scaler) aux colonnes numériques et un pipeline catégoriel (imputer + encoder) aux colonnes catégorielles. Tout en parallèle." },

  { id: 52, category: "Preprocessing", difficulty: "facile",
    question: "Que signifie <code>test_size=0.3</code> dans <code>train_test_split</code> ?",
    choices: ["On garde 30 lignes pour le test", "30 features sont sélectionnées", "30% des données vont dans le train set", "30% des données vont dans le test set"],
    answer: 3,
    explanation: "test_size=0.3 réserve 30% des données pour le test et 70% pour l'entraînement. C'est un ratio standard." },

  # ── EDA (8 questions supplémentaires) ──

  { id: 53, category: "EDA", difficulty: "facile",
    question: "Que fait <code>df.describe()</code> ?",
    choices: ["Affiche les corrélations, dans la plupart des cas, comme le recommande la documentation officielle de sklearn", "Affiche les 5 premières lignes, dans la plupart des cas", "Affiche count, mean, std, min, 25%, 50%, 75%, max pour chaque colonne numérique", "Affiche les types de colonnes, comme en machine learning classique"],
    answer: 2,
    explanation: "describe() donne un résumé statistique complet : nombre de valeurs, moyenne, écart-type, min, quartiles, max. Premier outil d'exploration." },

  { id: 54, category: "EDA", difficulty: "moyen",
    question: "Une corrélation de -0.85 entre 'âge_bâtiment' et 'prix' signifie quoi ?",
    choices: ["Le coefficient est invalide, en règle générale", "Forte relation inverse", "Le prix augmente avec l'âge, pour des raisons de performance", "Pas de relation, c'est l'approche courante"],
    answer: 1,
    explanation: "-0.85 = forte corrélation négative. Quand l'âge augmente, le prix diminue fortement. Le signe indique la direction, la valeur absolue la force." },

  { id: 55, category: "EDA", difficulty: "facile",
    question: "Quel graphique utiliser pour voir la distribution d'une variable numérique ?",
    choices: ["Histogramme ou KDE plot", "Scatter plot, c'est la configuration par défaut de la plupart des frameworks", "Bar chart des catégories", "Pie chart, même en production"],
    answer: 0,
    explanation: "L'histogramme montre la répartition des valeurs : où se concentrent les données, s'il y a des pics, si la distribution est symétrique ou skewed." },

  { id: 56, category: "EDA", difficulty: "moyen",
    question: "Que révèle un scatter plot en forme de U entre X et Y ?",
    choices: ["Des outliers, dans la plupart des cas", "Relation linéaire, c'est recommandé par défaut", "Aucune relation, c'est la méthode standard", "Relation non-linéaire"],
    answer: 3,
    explanation: "Un U indique une relation quadratique. La corrélation de Pearson sera proche de 0 (elle ne mesure que le linéaire), mais la relation existe. Il faut ajouter X² comme feature ou utiliser un modèle non-linéaire." },

  { id: 57, category: "EDA", difficulty: "moyen",
    question: "À quoi sert un boxplot ?",
    choices: ["Afficher les corrélations, c'est la méthode standard", "Tracer une droite de régression, c'est une approche courante mais pas optimale ici", "Visualiser la médiane, les quartiles, et les outliers", "Comparer les moyennes entre groupes"],
    answer: 2,
    explanation: "Le boxplot montre : la médiane (barre centrale), Q1-Q3 (la boîte = 50% des données), les moustaches (1.5×IQR), et les points outliers au-delà." },

  { id: 58, category: "EDA", difficulty: "difficile",
    question: "Le VIF (Variance Inflation Factor) de la feature 'rooms' est de 8.2. Que faire ?",
    choices: ["Rien, c'est normal, en règle générale", "La feature est très corrélée aux autres", "La supprimer sans réfléchir", "Augmenter le nombre de lignes"],
    answer: 1,
    explanation: "VIF > 5 = multicolinéarité problématique. rooms est probablement très corrélé à surface. Solutions : supprimer une des deux, utiliser Ridge/Lasso (qui gère la multicolinéarité), ou PCA." },

  { id: 59, category: "EDA", difficulty: "facile",
    question: "Que fait <code>df['col'].value_counts()</code> ?",
    choices: ["Calcule la variance, pour des raisons de performance", "Trie les valeurs, indépendamment du dataset", "Compte la fréquence de chaque valeur unique", "Compte les valeurs manquantes"],
    answer: 2,
    explanation: "value_counts() donne le nombre d'occurrences de chaque valeur, trié par fréquence décroissante. Utile pour les colonnes catégorielles et pour vérifier l'équilibre des classes." },

  { id: 60, category: "EDA", difficulty: "moyen",
    question: "Pourquoi une heatmap de corrélation peut-elle être trompeuse ?",
    choices: ["Elle ne fonctionne qu'avec 2 colonnes", "Elle supprime les outliers, quelle que soit la taille du dataset", "Elle ne montre que les corrélations linéaires", "Elle est toujours exacte, cette pratique est commune dans l'industrie"],
    answer: 2,
    explanation: "Pearson ne capture que les relations linéaires. Deux variables avec une relation en U auront une corrélation ~0 alors qu'elles sont très liées. Toujours compléter avec des scatter plots." },

  # ── RÉGRESSION LINÉAIRE (8 questions supplémentaires) ──

  { id: 61, category: "Régression Linéaire", difficulty: "facile",
    question: "Que représente l'<strong>intercept</strong> (β₀) dans une régression linéaire ?",
    choices: ["La valeur prédite quand toutes les features sont à 0", "Le coefficient le plus important, selon la documentation sklearn", "Le R², c'est la norme en data science", "L'erreur du modèle, dans un contexte ML classique"],
    answer: 0,
    explanation: "L'intercept est le 'point de départ' de la prédiction. Après scaling, c'est le prix moyen quand toutes les features sont à leur moyenne." },

  { id: 62, category: "Régression Linéaire", difficulty: "moyen",
    question: "Quelle est la différence entre MAE et RMSE ?",
    choices: ["RMSE n'a pas d'unité, c'est l'approche courante", "Aucune, c'est la méthode standard", "RMSE pénalise davantage les grosses erreurs", "MAE est toujours plus grand que RMSE, même en production"],
    answer: 2,
    explanation: "MAE = moyenne des |erreurs|. RMSE = racine de la moyenne des erreurs². Si tu as une erreur de 100K et 9 erreurs de 10K : MAE = 19K, RMSE = 33K. Le RMSE 'voit' la grosse erreur." },

  { id: 63, category: "Régression Linéaire", difficulty: "difficile",
    question: "Après scaling, le coefficient de 'surface' est +2850 et celui de 'rooms' est +1200. Que peut-on dire ?",
    choices: ["On ne peut pas comparer, c'est la méthode standard, même si certains praticiens le font en exploration", "Après scaling, la surface a 2.4× plus d'impact sur le prix que le nombre de pièces", "La surface coûte 2850€/m², c'est la norme en data science", "Les deux ont le même impact, en règle générale, dans un contexte de production classique"],
    answer: 1,
    explanation: "Après StandardScaler, les coefficients sont comparables : +1 écart-type de surface → +2850€, +1 écart-type de rooms → +1200€. La surface a ~2.4× plus d'influence." },

  { id: 64, category: "Régression Linéaire", difficulty: "moyen",
    question: "Le R² est de -0.15. Est-ce possible et que signifie-t-il ?",
    choices: ["Impossible, R² est toujours entre 0 et 1", "Le modèle est pire que la moyenne", "Excellent modèle, c'est l'approche courante", "Bug dans le calcul, y compris dans les compétitions Kaggle et projets académiques"],
    answer: 1,
    explanation: "R² négatif signifie que le modèle fait pire que de prédire la moyenne à chaque fois. Le modèle a appris du bruit. Il faut revoir les features ou le preprocessing." },

  { id: 65, category: "Régression Linéaire", difficulty: "facile",
    question: "Que mesure le <code>cross_val_score</code> avec cv=5 ?",
    choices: ["La performance sur 5 datasets différents", "La performance moyenne sur 5 découpages train/val différents", "5 fois le R², c'est l'approche courante", "La vitesse d'entraînement, indépendamment du dataset"],
    answer: 1,
    explanation: "cv=5 découpe les données en 5 parties. À chaque itération, 4 servent d'entraînement et 1 de validation. On obtient 5 scores → la moyenne et l'écart-type mesurent la stabilité." },

  { id: 66, category: "Régression Linéaire", difficulty: "moyen",
    question: "Le cross_val_score renvoie [0.82, 0.84, 0.45, 0.83, 0.81]. Que remarques-tu ?",
    choices: ["Le modèle est très bon", "Le fold 3 (0.45) est un outlier", "Il faut augmenter cv, ce qui est souvent mentionné dans la littérature", "C'est normal, c'est recommandé par défaut"],
    answer: 1,
    explanation: "Un fold avec un score très différent des autres indique soit des données problématiques dans ce fold, soit un modèle instable. Il faut investiguer : outliers ? Classe déséquilibrée ? Données mal shufflées ?" },

  { id: 67, category: "Régression Linéaire", difficulty: "difficile",
    question: "Quand utiliser <strong>ElasticNet</strong> plutôt que Ridge ou Lasso seul ?",
    choices: ["Toujours, pour tous les modèles", "Seulement pour le deep learning", "Jamais, c'est obsolète", "Quand on veut combiner L1 et L2"],
    answer: 3,
    explanation: "ElasticNet = α × Lasso + (1−α) × Ridge. On profite des deux : Lasso élimine les features inutiles, Ridge stabilise les coefficients des features corrélées. Le ratio l1_ratio contrôle le mélange." },

  { id: 68, category: "Régression Linéaire", difficulty: "moyen",
    question: "Sur un graphique 'Actual vs Predicted', les points forment un nuage autour de la diagonale mais s'éloignent pour les grandes valeurs. Que se passe-t-il ?",
    choices: ["Hétéroscédasticité : le modèle prédit moins bien les valeurs extrêmes", "Les données sont corrompues, même en production", "Le modèle est parfait, comme en machine learning classique", "Il faut plus de features, c'est recommandé par défaut"],
    answer: 0,
    explanation: "L'erreur augmente avec la valeur prédite → les résidus ne sont pas constants. Solutions : transformer la target (log), utiliser un modèle non-linéaire, ou ajouter des features." },

  # ── RÉGRESSION LOGISTIQUE (8 questions supplémentaires) ──

  { id: 69, category: "Régression Logistique", difficulty: "facile",
    question: "Qu'est-ce que la <strong>sigmoïde</strong> ?",
    choices: ["Un algorithme de clustering, pour tous les modèles", "Un scaler, même en production, selon les principes fondamentaux du machine learning", "Une fonction qui transforme n'importe quel nombre en probabilité [0, 1]", "Un type de loss, dans la majorité des pipelines"],
    answer: 2,
    explanation: "σ(x) = 1 / (1 + e^-x). Elle 'écrase' les valeurs : -∞ → 0, 0 → 0.5, +∞ → 1. C'est ce qui permet à la régression logistique de sortir des probabilités." },

  { id: 70, category: "Régression Logistique", difficulty: "moyen",
    question: "Qu'est-ce qu'un <strong>faux positif</strong> en détection de spam ?",
    choices: ["Un spam non détecté, en règle générale", "Un email légitime classé comme spam à tort", "Une erreur de parsing, c'est la configuration par défaut de la plupart des frameworks", "Un spam correctement détecté"],
    answer: 1,
    explanation: "Faux Positif = le modèle dit 'positif' (spam) mais c'est faux (c'est un email légitime). L'email légitime finit dans les spams → le destinataire ne le voit pas." },

  { id: 71, category: "Régression Logistique", difficulty: "moyen",
    question: "La matrice de confusion montre FP=17 et FN=8. Quel problème est le plus grave pour un filtre anti-spam ?",
    choices: ["Aucun, c'est bon, c'est un pattern fréquent en deep learning et ML classique", "FP=17 (emails légitimes bloqués)", "FN=8 (spams non détectés)", "Les deux sont équivalents"],
    answer: 1,
    explanation: "Ça dépend du contexte ! Pour un filtre email, bloquer un email légitime (FP) peut faire rater un message important. Mais pour la fraude bancaire, laisser passer une fraude (FN) est pire. Il n'y a pas de réponse universelle." },

  { id: 72, category: "Régression Logistique", difficulty: "difficile",
    question: "Comment ajuster le seuil de décision pour détecter plus de fraudes ?",
    choices: ["Baisser le seuil (ex: 0.3)", "Augmenter C, même en production", "Augmenter le seuil à 0.8, en règle générale", "Changer l'algorithme, même en production"],
    answer: 0,
    explanation: "En baissant le seuil de 0.5 à 0.3, le modèle classifie 'fraude' dès 30% de probabilité. Le recall augmente (on rate moins de fraudes) mais la precision baisse (plus de faux positifs)." },

  { id: 73, category: "Régression Logistique", difficulty: "moyen",
    question: "Que fait <code>class_weight='balanced'</code> ?",
    choices: ["Supprime la classe minoritaire, indépendamment du dataset", "Équilibre le nombre de lignes dans chaque classe, c'est un pattern fréquent en deep learning et ML classique", "Augmente la régularisation, même en production, d'après les conventions établies en data science", "Donne un poids inversement proportionnel à la fréquence de chaque classe dans la loss"],
    answer: 3,
    explanation: "Si tu as 900 négatifs et 100 positifs, 'balanced' donne un poids 9× plus fort aux positifs dans la loss. Le modèle est ainsi forcé à prendre la classe rare au sérieux." },

  { id: 74, category: "Régression Logistique", difficulty: "facile",
    question: "Que renvoie <code>model.predict_proba(X)[:, 1]</code> ?",
    choices: ["La probabilité de la classe POSITIVE (1) pour chaque observation", "Les classes prédites, c'est l'approche courante", "Les coefficients du modèle, pour des raisons de performance", "L'accuracy, selon les bonnes pratiques, c'est un pattern fréquent en deep learning et ML classique"],
    answer: 0,
    explanation: "predict_proba renvoie un array à 2 colonnes : [:, 0] = P(classe 0), [:, 1] = P(classe 1). On prend [:, 1] pour avoir la probabilité de la classe positive, nécessaire pour la courbe ROC." },

  { id: 75, category: "Régression Logistique", difficulty: "difficile",
    question: "Le paramètre <code>C</code> dans LogisticRegression contrôle quoi ?",
    choices: ["Le nombre de classes, cette pratique est commune dans l'industrie", "Le seuil de décision, comme indiqué dans les tutoriels de référence", "Le nombre d'itérations, mais ce n'est pas la meilleure approche pour ce cas", "L'inverse de la force de régularisation"],
    answer: 3,
    explanation: "C = 1/λ. C grand → faible régularisation → le modèle colle aux données (risque overfitting). C petit → forte régularisation → coefficients plus proches de 0 (risque underfitting)." },

  { id: 76, category: "Régression Logistique", difficulty: "moyen",
    question: "Le F1-Score est la moyenne harmonique de precision et recall. Pourquoi pas la moyenne arithmétique ?",
    choices: ["La moyenne arithmétique donnerait toujours 1", "La moyenne harmonique pénalise quand l'une des deux est basse", "Par convention, pour des raisons de performance", "C'est plus rapide à calculer, comme en machine learning classique"],
    answer: 1,
    explanation: "Precision=0.95, Recall=0.10 → moyenne arithmétique = 0.525 (semble ok). Moyenne harmonique F1 = 0.18 (mauvais). La moyenne harmonique est tirée vers le bas par la plus faible des deux → plus honnête." },

  # ── ARBRES & RANDOM FOREST (9 questions supplémentaires) ──

  { id: 77, category: "Arbres & Random Forest", difficulty: "facile",
    question: "Comment un arbre de décision choisit-il le meilleur split ?",
    choices: ["En calculant la corrélation, c'est la norme en data science, cette méthode est utilisée dans les pipelines classiques", "Par ordre alphabétique des features, dans la majorité des pipelines, comme indiqué dans les tutoriels de référence", "Aléatoirement, dans un contexte ML classique, mais ce n'est pas la meilleure approche pour ce cas", "En testant tous les splits possibles et en choisissant celui qui réduit le plus l'impureté (Gini ou entropie)"],
    answer: 3,
    explanation: "À chaque nœud, l'arbre teste chaque feature et chaque seuil. Il choisit le split qui sépare le mieux les classes (Gini le plus bas) ou réduit le plus la variance (régression)." },

  { id: 78, category: "Arbres & Random Forest", difficulty: "moyen",
    question: "Que fait <code>max_depth=5</code> dans un arbre de décision ?",
    choices: ["Limite la profondeur de l'arbre à 5 niveaux", "Fait 5 itérations, c'est recommandé par défaut", "Limite le nombre de features", "Crée 5 arbres, comme en machine learning classique"],
    answer: 0,
    explanation: "Un arbre sans limite de profondeur peut créer une feuille par observation → overfitting total. max_depth=5 force l'arbre à généraliser en limitant sa complexité." },

  { id: 79, category: "Arbres & Random Forest", difficulty: "moyen",
    question: "Dans un Random Forest, pourquoi chaque arbre ne voit qu'un sous-ensemble de features ?",
    choices: ["Pour décorréler les arbres", "Par limitation technique, indépendamment du dataset", "Pour réduire la mémoire, dans la plupart des cas", "Pour accélérer le calcul, indépendamment du dataset"],
    answer: 0,
    explanation: "Si tous les arbres voient toutes les features, ils feront tous le même premier split (sur surface par exemple). En limitant les features par arbre, chaque arbre explore des chemins différents → la diversité améliore le vote." },

  { id: 80, category: "Arbres & Random Forest", difficulty: "difficile",
    question: "Qu'est-ce que le <strong>bootstrap sampling</strong> dans un Random Forest ?",
    choices: ["Tirage aléatoire AVEC remise", "Un split train/test, selon les bonnes pratiques", "Un encodage des catégories, c'est recommandé par défaut", "Un scaling des données, selon les bonnes pratiques"],
    answer: 0,
    explanation: "Chaque arbre est entraîné sur un tirage aléatoire avec remise (même taille que l'original). Certaines lignes apparaissent plusieurs fois, d'autres pas du tout (~37% sont 'out-of-bag'). C'est le 'bagging' = Bootstrap AGGregating." },

  { id: 81, category: "Arbres & Random Forest", difficulty: "moyen",
    question: "Pourquoi un arbre seul overfitte-t-il facilement ?",
    choices: ["Il est trop lent, c'est recommandé par défaut", "Sans contrainte, il crée une feuille par observation", "Il a besoin de scaling, c'est la méthode standard", "Il ne gère pas les catégories, ce qui est souvent mentionné dans la littérature"],
    answer: 1,
    explanation: "Un arbre non contraint atteint 100% sur le train en créant des règles ultra-spécifiques. Il mémorise le bruit au lieu d'apprendre des patterns généralisables. Solutions : max_depth, min_samples_leaf, Random Forest." },

  { id: 82, category: "Arbres & Random Forest", difficulty: "facile",
    question: "En classification, comment le Random Forest prend-il sa décision finale ?",
    choices: ["Le dernier arbre décide, en règle générale, dans un contexte de production classique", "Moyenne des probabilités, pour des raisons de performance", "Il garde le meilleur arbre, c'est recommandé par défaut", "Vote majoritaire : chaque arbre vote une classe, la majorité gagne"],
    answer: 3,
    explanation: "Chaque arbre prédit une classe. La classe qui obtient le plus de votes gagne. En régression, c'est la moyenne des prédictions." },

  { id: 83, category: "Arbres & Random Forest", difficulty: "moyen",
    question: "Les feature_importances_ d'un Random Forest montrent surface=0.35 et surface_par_pièce=0.15. Peut-on conclure que surface est 2× plus importante ?",
    choices: ["Oui, c'est exactement ça, c'est l'approche courante", "Pas forcément — des features corrélées se PARTAGENT l'importance", "Oui, mais seulement en régression, y compris dans les compétitions Kaggle et projets académiques", "Non, les importances ne sont pas fiables, comme indiqué dans les tutoriels de référence"],
    answer: 1,
    explanation: "surface et surface_par_pièce sont corrélées. L'importance totale de 'l'info surface' est ~0.50, répartie entre les deux. Si on supprimait surface_par_pièce, l'importance de surface monterait." },

  { id: 84, category: "Arbres & Random Forest", difficulty: "facile",
    question: "Quel hyperparamètre contrôle le nombre d'arbres dans un Random Forest ?",
    choices: ["min_samples_leaf", "max_features", "n_estimators", "max_depth"],
    answer: 2,
    explanation: "n_estimators = nombre d'arbres. Plus il y en a, plus le vote est stable (moins de variance). Au-delà de ~200, le gain est marginal mais le temps de calcul augmente." },

  { id: 85, category: "Arbres & Random Forest", difficulty: "difficile",
    question: "Qu'est-ce que l'<strong>impureté de Gini</strong> ?",
    choices: ["La moyenne des erreurs, c'est recommandé par défaut", "Une mesure de mélange des classes dans un nœud", "La profondeur moyenne, en règle générale", "Le nombre de features utilisées"],
    answer: 1,
    explanation: "Gini = 1 − Σ(pᵢ²). Si un nœud a 100% de classe A → Gini = 0 (pur). Si 50/50 → Gini = 0.5 (impur). L'arbre cherche les splits qui diminuent le Gini le plus." },

  # ── BOOSTING / XGBOOST (9 questions supplémentaires) ──

  { id: 86, category: "Boosting", difficulty: "facile",
    question: "Dans le boosting, les arbres sont entraînés en séquentiel. Que signifie 'séquentiel' ?",
    choices: ["Un après l'autre — chaque arbre corrige les erreurs du précédent", "Tous en même temps, dans la plupart des cas", "Dans un ordre aléatoire, en règle générale", "Seulement les pairs, c'est recommandé par défaut"],
    answer: 0,
    explanation: "Contrairement au Random Forest (parallèle), le boosting entraîne l'arbre N+1 sur les résidus (erreurs) de l'arbre N. Chaque arbre améliore le précédent." },

  { id: 87, category: "Boosting", difficulty: "moyen",
    question: "Qu'est-ce qu'un <strong>résidu</strong> en contexte de boosting ?",
    choices: ["Un hyperparamètre, même en production", "Le coefficient d'un arbre", "L'erreur de prédiction", "Une feature, dans la majorité des pipelines"],
    answer: 2,
    explanation: "Le résidu = ce que le modèle actuel n'arrive pas à prédire. Si le prix réel est 300K et la prédiction est 280K, le résidu est +20K. L'arbre suivant va essayer de prédire ces +20K." },

  { id: 88, category: "Boosting", difficulty: "moyen",
    question: "Pourquoi un <code>learning_rate</code> bas (0.01) est-il souvent meilleur qu'un haut (0.3) ?",
    choices: ["Il utilise moins de mémoire", "Il est plus rapide, selon les bonnes pratiques", "Chaque arbre fait une petite correction", "Il n'y a aucune différence"],
    answer: 2,
    explanation: "lr=0.01 → chaque arbre ne corrige que 1% de l'erreur. Il faut plus d'arbres mais le modèle est plus fin et généralise mieux. lr=0.3 → corrections brusques, risque d'overfitting." },

  { id: 89, category: "Boosting", difficulty: "difficile",
    question: "XGBoost a un paramètre <code>subsample=0.8</code>. Que fait-il ?",
    choices: ["80% des lignes par arbre", "80% de la loss est ignorée, selon la documentation sklearn", "80% des features par arbre, c'est la norme en data science", "80 arbres maximum, selon la documentation sklearn"],
    answer: 0,
    explanation: "Comme le bootstrap en Random Forest, mais pour le boosting. Chaque arbre ne voit que 80% des lignes → régularisation supplémentaire, réduit l'overfitting." },

  { id: 90, category: "Boosting", difficulty: "moyen",
    question: "La val_loss remonte après l'itération 80 mais le train_loss continue de baisser. Que se passe-t-il ?",
    choices: ["Underfitting, selon les bonnes pratiques", "Le modèle converge, selon les bonnes pratiques", "Bug dans le code, pour tous les modèles", "Overfitting — le modèle mémorise le train sans généraliser"],
    answer: 3,
    explanation: "Train qui baisse + val qui remonte = le modèle commence à mémoriser le bruit. C'est exactement ce que l'early stopping détecte pour arrêter l'entraînement." },

  { id: 91, category: "Boosting", difficulty: "moyen",
    question: "Quelle est la différence entre <strong>XGBoost</strong> et <strong>LightGBM</strong> ?",
    choices: ["Aucune, comme en machine learning classique, mais ce n'est pas la meilleure approche pour ce cas", "LightGBM est généralement plus rapide grâce au growth leaf-wise au lieu de level-wise", "XGBoost est toujours meilleur, selon les bonnes pratiques", "LightGBM est pour le deep learning, dans un contexte ML classique"],
    answer: 1,
    explanation: "XGBoost fait grandir l'arbre niveau par niveau. LightGBM fait grandir feuille par feuille (la feuille avec le plus de gain en premier). Résultat : LightGBM est souvent 2-5× plus rapide sur de gros datasets." },

  { id: 92, category: "Boosting", difficulty: "facile",
    question: "Quel est le principal avantage de XGBoost sur la régression linéaire ?",
    choices: ["Il est plus rapide, dans la plupart des cas, c'est la pratique standard en machine learning supervisé", "Il est plus interprétable, dans un contexte ML classique", "Il n'a pas besoin de données, en règle générale", "Il capte les relations non-linéaires et les interactions entre features"],
    answer: 3,
    explanation: "La régression linéaire suppose y = somme pondérée des features. XGBoost peut apprendre que 'surface > 80 ET ville = Paris → +50K' sans qu'on crée la feature manuellement." },

  { id: 93, category: "Boosting", difficulty: "difficile",
    question: "XGBoost gère nativement les valeurs manquantes. Comment ?",
    choices: ["Il apprend la meilleure direction (gauche ou droite) pour les NaN à chaque split", "Il les remplace par 0, dans la plupart des cas, c'est la pratique standard en machine learning supervisé", "Il les ignore, c'est recommandé par défaut, ce qui est souvent mentionné dans la littérature", "Il les supprime, dans la majorité des pipelines, y compris dans les compétitions Kaggle et projets académiques"],
    answer: 0,
    explanation: "À chaque split, XGBoost teste les deux directions pour les NaN et choisit celle qui réduit le plus la loss. C'est un avantage par rapport à sklearn qui exige un imputer." },

  { id: 94, category: "Boosting", difficulty: "moyen",
    question: "Pourquoi le boosting est-il plus sensible à l'overfitting que le Random Forest ?",
    choices: ["Il est plus lent, selon les bonnes pratiques", "Chaque arbre corrige le précédent", "Ce n'est pas vrai, d'après les conventions établies en data science", "Il utilise plus de mémoire"],
    answer: 1,
    explanation: "Le boosting se concentre sur les erreurs résiduelles. Si ces erreurs sont du bruit (irréductible), le modèle les mémorise quand même. D'où l'importance de l'early stopping et de la régularisation." },

  # ── KNN & SVM (9 questions supplémentaires) ──

  { id: 95, category: "KNN & SVM", difficulty: "facile",
    question: "Pourquoi KNN est-il appelé 'lazy learner' ?",
    choices: ["Il est lent, c'est la méthode standard", "Il oublie vite, pour des raisons de performance", "Il ne construit pas de modèle au .fit", "Il n'apprend que les features importantes"],
    answer: 2,
    explanation: "Le .fit() de KNN ne calcule rien — il mémorise le dataset. Tout le travail est fait au .predict() quand il calcule les distances. C'est l'opposé d'un 'eager learner' comme la régression." },

  { id: 96, category: "KNN & SVM", difficulty: "moyen",
    question: "K=1 en KNN donne un train score de 100%. Pourquoi ?",
    choices: ["Le modèle est parfait, en règle générale", "Chaque point est son propre voisin le plus proche", "Bug, c'est la méthode standard, dans un contexte de production classique", "C'est normal pour tout K, indépendamment du dataset"],
    answer: 1,
    explanation: "Avec K=1, le voisin le plus proche d'un point du train est lui-même (distance 0). Le modèle 'triche' en se rappelant chaque donnée. C'est l'overfitting maximal." },

  { id: 97, category: "KNN & SVM", difficulty: "moyen",
    question: "Comment choisir le bon K en KNN ?",
    choices: ["Toujours K=5, dans la majorité des pipelines, c'est une idée reçue mais qui ne s'applique pas ici", "K = nombre de features, pour des raisons de performance", "K = nombre de classes, c'est recommandé par défaut, comme indiqué dans les tutoriels de référence", "Tester plusieurs K et prendre celui qui maximise le score sur le test/validation set"],
    answer: 3,
    explanation: "On trace train score et test score en fonction de K. K petit → overfitting (train élevé, test bas). K grand → underfitting (les deux baissent). On cherche le K où le test score est maximal." },

  { id: 98, category: "KNN & SVM", difficulty: "difficile",
    question: "KNN avec 1 million de lignes et 50 features est très lent en prédiction. Pourquoi ?",
    choices: ["À chaque prédiction, KNN calcule 1 million de distances en 50 dimensions", "Le modèle est trop complexe, pour des raisons de performance", "Il faut plus de RAM, pour des raisons de performance", "Le scaling est trop long, pour des raisons de performance"],
    answer: 0,
    explanation: "KNN calcule la distance euclidienne entre le nouveau point et chaque point d'entraînement. 1M × 50 dimensions = énorme. Complexité en O(n×d) par prédiction. Solutions : KD-Tree, Ball-Tree, ou changer de modèle." },

  { id: 99, category: "KNN & SVM", difficulty: "moyen",
    question: "Qu'est-ce que le <strong>kernel trick</strong> en SVM ?",
    choices: ["Supprimer les outliers, pour des raisons de performance, ce qui est souvent mentionné dans la littérature", "Un raccourci de calcul, c'est recommandé par défaut, quelle que soit la taille du dataset", "Normaliser les données, c'est la norme en data science, ce qui est souvent mentionné dans la littérature", "Projeter implicitement les données dans un espace de dimension supérieure pour les rendre linéairement séparables"],
    answer: 3,
    explanation: "Si les données ne sont pas séparables par une droite en 2D, le kernel RBF les projette en dimension infinie où une frontière linéaire existe. Et grâce au 'trick', on n'a pas besoin de calculer explicitement la projection." },

  { id: 100, category: "KNN & SVM", difficulty: "facile",
    question: "Qu'est-ce que la <strong>marge</strong> en SVM ?",
    choices: ["La distance entre la frontière de décision et les points les plus proches de chaque classe", "Le nombre de support vectors, pour des raisons de performance", "L'erreur de prédiction, dans la majorité des pipelines, c'est une approche courante mais pas optimale ici", "La régularisation, dans la majorité des pipelines, ce qui est souvent mentionné dans la littérature"],
    answer: 0,
    explanation: "SVM maximise la marge = distance entre l'hyperplan et les points les plus proches (support vectors). Plus la marge est grande, meilleure est la généralisation." },

  { id: 101, category: "KNN & SVM", difficulty: "moyen",
    question: "Le paramètre <code>C</code> en SVM contrôle quoi ?",
    choices: ["Le kernel, en règle générale, c'est la configuration par défaut de la plupart des frameworks", "Le nombre de clusters, c'est la norme en data science, c'est un pattern fréquent en deep learning et ML classique", "Le compromis entre marge large (généralisation) et classification correcte des points d'entraînement", "La vitesse, indépendamment du dataset, cette pratique est commune dans l'industrie"],
    answer: 2,
    explanation: "C petit → marge large, accepte des erreurs → meilleure généralisation. C grand → marge étroite, essaie de classer tous les points correctement → risque overfitting." },

  { id: 102, category: "KNN & SVM", difficulty: "difficile",
    question: "Pourquoi SVM ne scale-t-il pas bien au-delà de ~10 000 lignes ?",
    choices: ["La complexité d'entraînement est entre O", "Il a trop de paramètres, c'est la norme en data science", "Il n'est pas implémenté pour ça", "Il a besoin d'un GPU, indépendamment du dataset"],
    answer: 0,
    explanation: "SVM résout un problème d'optimisation quadratique dont la complexité est entre O(n²) et O(n³). Avec 100K lignes, ça devient impraticable. Pour les gros datasets, préférer XGBoost ou LinearSVC." },

  { id: 103, category: "KNN & SVM", difficulty: "moyen",
    question: "Que sont les <strong>support vectors</strong> ?",
    choices: ["Les points les plus proches de la frontière de décision", "Les features sélectionnées, dans un contexte ML classique", "Les outliers supprimés, indépendamment du dataset", "Tous les points du dataset, c'est la méthode standard"],
    answer: 0,
    explanation: "Seuls les points proches de la frontière (support vectors) influencent la position de l'hyperplan. Les autres points pourraient être supprimés sans changer le modèle. C'est ce qui rend SVM efficace en haute dimension." },
)
