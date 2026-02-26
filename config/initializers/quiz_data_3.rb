# frozen_string_literal: true
# Additional quiz questions (41-200) appended to QUIZ_QUESTIONS

QUIZ_QUESTIONS.push(

  # ── PREPROCESSING (12 questions supplémentaires) ──

  { id: 41, category: "Preprocessing", difficulty: "facile",
    question: "Que fait <code>df.isna().sum()</code> ?",
    choices: ["Compte les lignes du DataFrame", "Compte les valeurs manquantes par colonne", "Remplace les NaN par 0", "Supprime les NaN"],
    answer: 1,
    explanation: "isna() crée un masque True/False pour chaque cellule, puis sum() compte les True par colonne. Premier réflexe pour diagnostiquer les données manquantes." },

  { id: 42, category: "Preprocessing", difficulty: "moyen",
    question: "Pourquoi séparer <code>train_test_split</code> AVANT le preprocessing ?",
    choices: ["Pour gagner du temps de calcul", "Pour éviter le data leakage — le test set doit rester invisible", "Parce que sklearn l'exige", "Pour avoir des sets de taille égale"],
    answer: 1,
    explanation: "Si tu normalises AVANT le split, le scaler voit les données de test → le modèle a indirectement 'vu' le test. C'est du data leakage. Toujours : split → fit(train) → transform(test)." },

  { id: 43, category: "Preprocessing", difficulty: "difficile",
    question: "Quelle est la différence entre <code>StandardScaler</code> et <code>MinMaxScaler</code> ?",
    choices: ["Aucune différence", "StandardScaler centre à μ=0, σ=1 ; MinMaxScaler ramène entre 0 et 1", "MinMaxScaler est plus rapide", "StandardScaler ne fonctionne qu'avec des entiers"],
    answer: 1,
    explanation: "StandardScaler : (x − μ) / σ → distribution centrée réduite. MinMaxScaler : (x − min) / (max − min) → entre 0 et 1. StandardScaler est préféré quand il y a des outliers (moins sensible aux extrêmes)." },

  { id: 44, category: "Preprocessing", difficulty: "moyen",
    question: "Tu as une colonne <code>taille</code> avec les valeurs S, M, L, XL. Quel encodage utiliser ?",
    choices: ["One-Hot Encoding", "Ordinal Encoding car il y a un ordre naturel", "Label Encoding aléatoire", "Supprimer la colonne"],
    answer: 1,
    explanation: "S < M < L < XL → il y a un ordre. Ordinal Encoding (S=1, M=2, L=3, XL=4) préserve cette hiérarchie. One-Hot perdrait l'information d'ordre." },

  { id: 45, category: "Preprocessing", difficulty: "facile",
    question: "Que fait <code>df.drop_duplicates()</code> ?",
    choices: ["Supprime les colonnes en double", "Supprime les lignes entièrement identiques", "Supprime les valeurs NaN", "Supprime les outliers"],
    answer: 1,
    explanation: "Retire les lignes où TOUTES les valeurs sont identiques à une autre ligne. Avec subset=['col1', 'col2'], on peut ne vérifier que certaines colonnes." },

  { id: 46, category: "Preprocessing", difficulty: "difficile",
    question: "Quelle méthode détecte les outliers avec l'IQR ?",
    choices: ["Supprimer les valeurs > moyenne", "Valeurs hors de [Q1 − 1.5×IQR, Q3 + 1.5×IQR]", "Toute valeur > 2 écarts-types", "Les valeurs NaN sont des outliers"],
    answer: 1,
    explanation: "IQR = Q3 − Q1. Les valeurs en dessous de Q1 − 1.5×IQR ou au-dessus de Q3 + 1.5×IQR sont considérées comme outliers. C'est la méthode des boxplots." },

  { id: 47, category: "Preprocessing", difficulty: "moyen",
    question: "Pourquoi imputer la target (y) manquante est-il une mauvaise idée ?",
    choices: ["C'est trop lent", "On inventerait des réponses fausses — mieux vaut supprimer ces lignes", "La target n'a jamais de NaN", "C'est une bonne idée en fait"],
    answer: 1,
    explanation: "Imputer la target = inventer la réponse. Le modèle apprendrait sur des données fictives. Toujours supprimer les lignes où la target est manquante." },

  { id: 48, category: "Preprocessing", difficulty: "facile",
    question: "Que renvoie <code>df.dtypes</code> ?",
    choices: ["Les statistiques descriptives", "Le type de chaque colonne (int64, float64, object...)", "Le nombre de valeurs uniques", "Les corrélations"],
    answer: 1,
    explanation: "dtypes montre le type de chaque colonne. 'object' = souvent du texte (catégoriel), 'int64'/'float64' = numérique. Essentiel pour savoir quoi encoder." },

  { id: 49, category: "Preprocessing", difficulty: "moyen",
    question: "Tu as 200 catégories dans une colonne 'ville'. One-Hot Encoding est-il une bonne idée ?",
    choices: ["Oui, toujours", "Non — 200 colonnes binaires = explosion dimensionnelle", "Oui mais seulement en régression", "Ça dépend du processeur"],
    answer: 1,
    explanation: "One-Hot sur 200 villes crée 200 colonnes. La matrice devient très sparse, le modèle ralentit et peut overfitter. Alternatives : Target Encoding, regrouper les villes rares, ou embeddings." },

  { id: 50, category: "Preprocessing", difficulty: "difficile",
    question: "Que fait <code>SimpleImputer(strategy='most_frequent')</code> ?",
    choices: ["Remplace les NaN par la moyenne", "Remplace les NaN par la valeur la plus fréquente (mode)", "Supprime les lignes avec NaN", "Remplace les NaN par 0"],
    answer: 1,
    explanation: "Le mode (valeur la plus fréquente) est la stratégie par défaut pour les colonnes catégorielles. Pour les numériques, on préfère 'median' (robuste aux outliers) ou 'mean'." },

  { id: 51, category: "Preprocessing", difficulty: "moyen",
    question: "À quoi sert <code>ColumnTransformer</code> ?",
    choices: ["Transformer toutes les colonnes de la même façon", "Appliquer des transformations différentes selon le type de colonne", "Supprimer les colonnes inutiles", "Renommer les colonnes"],
    answer: 1,
    explanation: "ColumnTransformer applique un pipeline numérique (imputer + scaler) aux colonnes numériques et un pipeline catégoriel (imputer + encoder) aux colonnes catégorielles. Tout en parallèle." },

  { id: 52, category: "Preprocessing", difficulty: "facile",
    question: "Que signifie <code>test_size=0.3</code> dans <code>train_test_split</code> ?",
    choices: ["30% des données vont dans le train set", "30% des données vont dans le test set", "On garde 30 lignes pour le test", "30 features sont sélectionnées"],
    answer: 1,
    explanation: "test_size=0.3 réserve 30% des données pour le test et 70% pour l'entraînement. C'est un ratio standard." },

  # ── EDA (8 questions supplémentaires) ──

  { id: 53, category: "EDA", difficulty: "facile",
    question: "Que fait <code>df.describe()</code> ?",
    choices: ["Affiche les 5 premières lignes", "Affiche count, mean, std, min, 25%, 50%, 75%, max pour chaque colonne numérique", "Affiche les types de colonnes", "Affiche les corrélations"],
    answer: 1,
    explanation: "describe() donne un résumé statistique complet : nombre de valeurs, moyenne, écart-type, min, quartiles, max. Premier outil d'exploration." },

  { id: 54, category: "EDA", difficulty: "moyen",
    question: "Une corrélation de -0.85 entre 'âge_bâtiment' et 'prix' signifie quoi ?",
    choices: ["Pas de relation", "Forte relation inverse : plus c'est vieux, moins c'est cher", "Le prix augmente avec l'âge", "Le coefficient est invalide"],
    answer: 1,
    explanation: "-0.85 = forte corrélation négative. Quand l'âge augmente, le prix diminue fortement. Le signe indique la direction, la valeur absolue la force." },

  { id: 55, category: "EDA", difficulty: "facile",
    question: "Quel graphique utiliser pour voir la distribution d'une variable numérique ?",
    choices: ["Scatter plot", "Histogramme ou KDE plot", "Bar chart des catégories", "Pie chart"],
    answer: 1,
    explanation: "L'histogramme montre la répartition des valeurs : où se concentrent les données, s'il y a des pics, si la distribution est symétrique ou skewed." },

  { id: 56, category: "EDA", difficulty: "moyen",
    question: "Que révèle un scatter plot en forme de U entre X et Y ?",
    choices: ["Relation linéaire", "Relation non-linéaire — une régression linéaire sera mauvaise", "Aucune relation", "Des outliers"],
    answer: 1,
    explanation: "Un U indique une relation quadratique. La corrélation de Pearson sera proche de 0 (elle ne mesure que le linéaire), mais la relation existe. Il faut ajouter X² comme feature ou utiliser un modèle non-linéaire." },

  { id: 57, category: "EDA", difficulty: "moyen",
    question: "À quoi sert un boxplot ?",
    choices: ["Comparer les moyennes entre groupes", "Visualiser la médiane, les quartiles, et les outliers", "Tracer une droite de régression", "Afficher les corrélations"],
    answer: 1,
    explanation: "Le boxplot montre : la médiane (barre centrale), Q1-Q3 (la boîte = 50% des données), les moustaches (1.5×IQR), et les points outliers au-delà." },

  { id: 58, category: "EDA", difficulty: "difficile",
    question: "Le VIF (Variance Inflation Factor) de la feature 'rooms' est de 8.2. Que faire ?",
    choices: ["Rien, c'est normal", "La feature est très corrélée aux autres — risque de multicolinéarité", "La supprimer sans réfléchir", "Augmenter le nombre de lignes"],
    answer: 1,
    explanation: "VIF > 5 = multicolinéarité problématique. rooms est probablement très corrélé à surface. Solutions : supprimer une des deux, utiliser Ridge/Lasso (qui gère la multicolinéarité), ou PCA." },

  { id: 59, category: "EDA", difficulty: "facile",
    question: "Que fait <code>df['col'].value_counts()</code> ?",
    choices: ["Compte les valeurs manquantes", "Compte la fréquence de chaque valeur unique", "Calcule la variance", "Trie les valeurs"],
    answer: 1,
    explanation: "value_counts() donne le nombre d'occurrences de chaque valeur, trié par fréquence décroissante. Utile pour les colonnes catégorielles et pour vérifier l'équilibre des classes." },

  { id: 60, category: "EDA", difficulty: "moyen",
    question: "Pourquoi une heatmap de corrélation peut-elle être trompeuse ?",
    choices: ["Elle ne montre que les corrélations linéaires", "Elle est toujours exacte", "Elle ne fonctionne qu'avec 2 colonnes", "Elle supprime les outliers"],
    answer: 0,
    explanation: "Pearson ne capture que les relations linéaires. Deux variables avec une relation en U auront une corrélation ~0 alors qu'elles sont très liées. Toujours compléter avec des scatter plots." },

  # ── RÉGRESSION LINÉAIRE (8 questions supplémentaires) ──

  { id: 61, category: "Régression Linéaire", difficulty: "facile",
    question: "Que représente l'<strong>intercept</strong> (β₀) dans une régression linéaire ?",
    choices: ["Le coefficient le plus important", "La valeur prédite quand toutes les features sont à 0 (ou à leur moyenne après scaling)", "L'erreur du modèle", "Le R²"],
    answer: 1,
    explanation: "L'intercept est le 'point de départ' de la prédiction. Après scaling, c'est le prix moyen quand toutes les features sont à leur moyenne." },

  { id: 62, category: "Régression Linéaire", difficulty: "moyen",
    question: "Quelle est la différence entre MAE et RMSE ?",
    choices: ["Aucune", "RMSE pénalise davantage les grosses erreurs (car elles sont élevées au carré)", "MAE est toujours plus grand que RMSE", "RMSE n'a pas d'unité"],
    answer: 1,
    explanation: "MAE = moyenne des |erreurs|. RMSE = racine de la moyenne des erreurs². Si tu as une erreur de 100K et 9 erreurs de 10K : MAE = 19K, RMSE = 33K. Le RMSE 'voit' la grosse erreur." },

  { id: 63, category: "Régression Linéaire", difficulty: "difficile",
    question: "Après scaling, le coefficient de 'surface' est +2850 et celui de 'rooms' est +1200. Que peut-on dire ?",
    choices: ["La surface coûte 2850€/m²", "Après scaling, la surface a 2.4× plus d'impact sur le prix que le nombre de pièces", "Les deux ont le même impact", "On ne peut pas comparer"],
    answer: 1,
    explanation: "Après StandardScaler, les coefficients sont comparables : +1 écart-type de surface → +2850€, +1 écart-type de rooms → +1200€. La surface a ~2.4× plus d'influence." },

  { id: 64, category: "Régression Linéaire", difficulty: "moyen",
    question: "Le R² est de -0.15. Est-ce possible et que signifie-t-il ?",
    choices: ["Impossible, R² est toujours entre 0 et 1", "Le modèle est pire que la moyenne — il prédit moins bien qu'une constante", "Excellent modèle", "Bug dans le calcul"],
    answer: 1,
    explanation: "R² négatif signifie que le modèle fait pire que de prédire la moyenne à chaque fois. Le modèle a appris du bruit. Il faut revoir les features ou le preprocessing." },

  { id: 65, category: "Régression Linéaire", difficulty: "facile",
    question: "Que mesure le <code>cross_val_score</code> avec cv=5 ?",
    choices: ["La performance sur 5 datasets différents", "La performance moyenne sur 5 découpages train/val différents", "5 fois le R²", "La vitesse d'entraînement"],
    answer: 1,
    explanation: "cv=5 découpe les données en 5 parties. À chaque itération, 4 servent d'entraînement et 1 de validation. On obtient 5 scores → la moyenne et l'écart-type mesurent la stabilité." },

  { id: 66, category: "Régression Linéaire", difficulty: "moyen",
    question: "Le cross_val_score renvoie [0.82, 0.84, 0.45, 0.83, 0.81]. Que remarques-tu ?",
    choices: ["Le modèle est très bon", "Le fold 3 (0.45) est un outlier → possible problème de données dans ce fold", "C'est normal", "Il faut augmenter cv"],
    answer: 1,
    explanation: "Un fold avec un score très différent des autres indique soit des données problématiques dans ce fold, soit un modèle instable. Il faut investiguer : outliers ? Classe déséquilibrée ? Données mal shufflées ?" },

  { id: 67, category: "Régression Linéaire", difficulty: "difficile",
    question: "Quand utiliser <strong>ElasticNet</strong> plutôt que Ridge ou Lasso seul ?",
    choices: ["Toujours", "Quand on veut combiner L1 et L2 — sélection de features + réduction des coefficients", "Jamais, c'est obsolète", "Seulement pour le deep learning"],
    answer: 1,
    explanation: "ElasticNet = α × Lasso + (1−α) × Ridge. On profite des deux : Lasso élimine les features inutiles, Ridge stabilise les coefficients des features corrélées. Le ratio l1_ratio contrôle le mélange." },

  { id: 68, category: "Régression Linéaire", difficulty: "moyen",
    question: "Sur un graphique 'Actual vs Predicted', les points forment un nuage autour de la diagonale mais s'éloignent pour les grandes valeurs. Que se passe-t-il ?",
    choices: ["Le modèle est parfait", "Hétéroscédasticité : le modèle prédit moins bien les valeurs extrêmes", "Les données sont corrompues", "Il faut plus de features"],
    answer: 1,
    explanation: "L'erreur augmente avec la valeur prédite → les résidus ne sont pas constants. Solutions : transformer la target (log), utiliser un modèle non-linéaire, ou ajouter des features." },

  # ── RÉGRESSION LOGISTIQUE (8 questions supplémentaires) ──

  { id: 69, category: "Régression Logistique", difficulty: "facile",
    question: "Qu'est-ce que la <strong>sigmoïde</strong> ?",
    choices: ["Un type de loss", "Une fonction qui transforme n'importe quel nombre en probabilité [0, 1]", "Un algorithme de clustering", "Un scaler"],
    answer: 1,
    explanation: "σ(x) = 1 / (1 + e^-x). Elle 'écrase' les valeurs : -∞ → 0, 0 → 0.5, +∞ → 1. C'est ce qui permet à la régression logistique de sortir des probabilités." },

  { id: 70, category: "Régression Logistique", difficulty: "moyen",
    question: "Qu'est-ce qu'un <strong>faux positif</strong> en détection de spam ?",
    choices: ["Un spam non détecté", "Un email légitime classé comme spam à tort", "Un spam correctement détecté", "Une erreur de parsing"],
    answer: 1,
    explanation: "Faux Positif = le modèle dit 'positif' (spam) mais c'est faux (c'est un email légitime). L'email légitime finit dans les spams → le destinataire ne le voit pas." },

  { id: 71, category: "Régression Logistique", difficulty: "moyen",
    question: "La matrice de confusion montre FP=17 et FN=8. Quel problème est le plus grave pour un filtre anti-spam ?",
    choices: ["FP=17 (emails légitimes bloqués)", "FN=8 (spams non détectés)", "Les deux sont équivalents", "Aucun, c'est bon"],
    answer: 0,
    explanation: "Ça dépend du contexte ! Pour un filtre email, bloquer un email légitime (FP) peut faire rater un message important. Mais pour la fraude bancaire, laisser passer une fraude (FN) est pire. Il n'y a pas de réponse universelle." },

  { id: 72, category: "Régression Logistique", difficulty: "difficile",
    question: "Comment ajuster le seuil de décision pour détecter plus de fraudes ?",
    choices: ["Augmenter le seuil à 0.8", "Baisser le seuil (ex: 0.3) → plus de détection mais plus de faux positifs", "Changer l'algorithme", "Augmenter C"],
    answer: 1,
    explanation: "En baissant le seuil de 0.5 à 0.3, le modèle classifie 'fraude' dès 30% de probabilité. Le recall augmente (on rate moins de fraudes) mais la precision baisse (plus de faux positifs)." },

  { id: 73, category: "Régression Logistique", difficulty: "moyen",
    question: "Que fait <code>class_weight='balanced'</code> ?",
    choices: ["Équilibre le nombre de lignes dans chaque classe", "Donne un poids inversement proportionnel à la fréquence de chaque classe dans la loss", "Supprime la classe minoritaire", "Augmente la régularisation"],
    answer: 1,
    explanation: "Si tu as 900 négatifs et 100 positifs, 'balanced' donne un poids 9× plus fort aux positifs dans la loss. Le modèle est ainsi forcé à prendre la classe rare au sérieux." },

  { id: 74, category: "Régression Logistique", difficulty: "facile",
    question: "Que renvoie <code>model.predict_proba(X)[:, 1]</code> ?",
    choices: ["Les classes prédites", "La probabilité de la classe POSITIVE (1) pour chaque observation", "Les coefficients du modèle", "L'accuracy"],
    answer: 1,
    explanation: "predict_proba renvoie un array à 2 colonnes : [:, 0] = P(classe 0), [:, 1] = P(classe 1). On prend [:, 1] pour avoir la probabilité de la classe positive, nécessaire pour la courbe ROC." },

  { id: 75, category: "Régression Logistique", difficulty: "difficile",
    question: "Le paramètre <code>C</code> dans LogisticRegression contrôle quoi ?",
    choices: ["Le nombre de classes", "L'inverse de la force de régularisation — C grand = moins de régularisation", "Le seuil de décision", "Le nombre d'itérations"],
    answer: 1,
    explanation: "C = 1/λ. C grand → faible régularisation → le modèle colle aux données (risque overfitting). C petit → forte régularisation → coefficients plus proches de 0 (risque underfitting)." },

  { id: 76, category: "Régression Logistique", difficulty: "moyen",
    question: "Le F1-Score est la moyenne harmonique de precision et recall. Pourquoi pas la moyenne arithmétique ?",
    choices: ["Par convention", "La moyenne harmonique pénalise quand l'une des deux est basse — elle force l'équilibre", "La moyenne arithmétique donnerait toujours 1", "C'est plus rapide à calculer"],
    answer: 1,
    explanation: "Precision=0.95, Recall=0.10 → moyenne arithmétique = 0.525 (semble ok). Moyenne harmonique F1 = 0.18 (mauvais). La moyenne harmonique est tirée vers le bas par la plus faible des deux → plus honnête." },

  # ── ARBRES & RANDOM FOREST (9 questions supplémentaires) ──

  { id: 77, category: "Arbres & Random Forest", difficulty: "facile",
    question: "Comment un arbre de décision choisit-il le meilleur split ?",
    choices: ["Aléatoirement", "En testant tous les splits possibles et en choisissant celui qui réduit le plus l'impureté (Gini ou entropie)", "En calculant la corrélation", "Par ordre alphabétique des features"],
    answer: 1,
    explanation: "À chaque nœud, l'arbre teste chaque feature et chaque seuil. Il choisit le split qui sépare le mieux les classes (Gini le plus bas) ou réduit le plus la variance (régression)." },

  { id: 78, category: "Arbres & Random Forest", difficulty: "moyen",
    question: "Que fait <code>max_depth=5</code> dans un arbre de décision ?",
    choices: ["Limite le nombre de features", "Limite la profondeur de l'arbre à 5 niveaux → régularisation", "Crée 5 arbres", "Fait 5 itérations"],
    answer: 1,
    explanation: "Un arbre sans limite de profondeur peut créer une feuille par observation → overfitting total. max_depth=5 force l'arbre à généraliser en limitant sa complexité." },

  { id: 79, category: "Arbres & Random Forest", difficulty: "moyen",
    question: "Dans un Random Forest, pourquoi chaque arbre ne voit qu'un sous-ensemble de features ?",
    choices: ["Pour accélérer le calcul", "Pour décorréler les arbres — sinon ils feraient tous le même split sur la feature dominante", "Par limitation technique", "Pour réduire la mémoire"],
    answer: 1,
    explanation: "Si tous les arbres voient toutes les features, ils feront tous le même premier split (sur surface par exemple). En limitant les features par arbre, chaque arbre explore des chemins différents → la diversité améliore le vote." },

  { id: 80, category: "Arbres & Random Forest", difficulty: "difficile",
    question: "Qu'est-ce que le <strong>bootstrap sampling</strong> dans un Random Forest ?",
    choices: ["Un scaling des données", "Tirage aléatoire AVEC remise — chaque arbre voit un échantillon différent", "Un split train/test", "Un encodage des catégories"],
    answer: 1,
    explanation: "Chaque arbre est entraîné sur un tirage aléatoire avec remise (même taille que l'original). Certaines lignes apparaissent plusieurs fois, d'autres pas du tout (~37% sont 'out-of-bag'). C'est le 'bagging' = Bootstrap AGGregating." },

  { id: 81, category: "Arbres & Random Forest", difficulty: "moyen",
    question: "Pourquoi un arbre seul overfitte-t-il facilement ?",
    choices: ["Il est trop lent", "Sans contrainte, il crée une feuille par observation → mémorise le bruit", "Il ne gère pas les catégories", "Il a besoin de scaling"],
    answer: 1,
    explanation: "Un arbre non contraint atteint 100% sur le train en créant des règles ultra-spécifiques. Il mémorise le bruit au lieu d'apprendre des patterns généralisables. Solutions : max_depth, min_samples_leaf, Random Forest." },

  { id: 82, category: "Arbres & Random Forest", difficulty: "facile",
    question: "En classification, comment le Random Forest prend-il sa décision finale ?",
    choices: ["Il garde le meilleur arbre", "Vote majoritaire : chaque arbre vote une classe, la majorité gagne", "Moyenne des probabilités", "Le dernier arbre décide"],
    answer: 1,
    explanation: "Chaque arbre prédit une classe. La classe qui obtient le plus de votes gagne. En régression, c'est la moyenne des prédictions." },

  { id: 83, category: "Arbres & Random Forest", difficulty: "moyen",
    question: "Les feature_importances_ d'un Random Forest montrent surface=0.35 et surface_par_pièce=0.15. Peut-on conclure que surface est 2× plus importante ?",
    choices: ["Oui, c'est exactement ça", "Pas forcément — des features corrélées se PARTAGENT l'importance", "Non, les importances ne sont pas fiables", "Oui, mais seulement en régression"],
    answer: 1,
    explanation: "surface et surface_par_pièce sont corrélées. L'importance totale de 'l'info surface' est ~0.50, répartie entre les deux. Si on supprimait surface_par_pièce, l'importance de surface monterait." },

  { id: 84, category: "Arbres & Random Forest", difficulty: "facile",
    question: "Quel hyperparamètre contrôle le nombre d'arbres dans un Random Forest ?",
    choices: ["max_depth", "n_estimators", "max_features", "min_samples_leaf"],
    answer: 1,
    explanation: "n_estimators = nombre d'arbres. Plus il y en a, plus le vote est stable (moins de variance). Au-delà de ~200, le gain est marginal mais le temps de calcul augmente." },

  { id: 85, category: "Arbres & Random Forest", difficulty: "difficile",
    question: "Qu'est-ce que l'<strong>impureté de Gini</strong> ?",
    choices: ["La moyenne des erreurs", "Une mesure de mélange des classes dans un nœud — 0 = pur, 0.5 = mélange maximal (binaire)", "Le nombre de features utilisées", "La profondeur moyenne"],
    answer: 1,
    explanation: "Gini = 1 − Σ(pᵢ²). Si un nœud a 100% de classe A → Gini = 0 (pur). Si 50/50 → Gini = 0.5 (impur). L'arbre cherche les splits qui diminuent le Gini le plus." },

  # ── BOOSTING / XGBOOST (9 questions supplémentaires) ──

  { id: 86, category: "Boosting", difficulty: "facile",
    question: "Dans le boosting, les arbres sont entraînés en séquentiel. Que signifie 'séquentiel' ?",
    choices: ["Tous en même temps", "Un après l'autre — chaque arbre corrige les erreurs du précédent", "Dans un ordre aléatoire", "Seulement les pairs"],
    answer: 1,
    explanation: "Contrairement au Random Forest (parallèle), le boosting entraîne l'arbre N+1 sur les résidus (erreurs) de l'arbre N. Chaque arbre améliore le précédent." },

  { id: 87, category: "Boosting", difficulty: "moyen",
    question: "Qu'est-ce qu'un <strong>résidu</strong> en contexte de boosting ?",
    choices: ["Le coefficient d'un arbre", "L'erreur de prédiction : y_réel − y_prédit", "Un hyperparamètre", "Une feature"],
    answer: 1,
    explanation: "Le résidu = ce que le modèle actuel n'arrive pas à prédire. Si le prix réel est 300K et la prédiction est 280K, le résidu est +20K. L'arbre suivant va essayer de prédire ces +20K." },

  { id: 88, category: "Boosting", difficulty: "moyen",
    question: "Pourquoi un <code>learning_rate</code> bas (0.01) est-il souvent meilleur qu'un haut (0.3) ?",
    choices: ["Il est plus rapide", "Chaque arbre fait une petite correction → le modèle apprend plus finement, moins d'overfitting", "Il utilise moins de mémoire", "Il n'y a aucune différence"],
    answer: 1,
    explanation: "lr=0.01 → chaque arbre ne corrige que 1% de l'erreur. Il faut plus d'arbres mais le modèle est plus fin et généralise mieux. lr=0.3 → corrections brusques, risque d'overfitting." },

  { id: 89, category: "Boosting", difficulty: "difficile",
    question: "XGBoost a un paramètre <code>subsample=0.8</code>. Que fait-il ?",
    choices: ["80% des features par arbre", "80% des lignes par arbre (échantillonnage stochastique)", "80% de la loss est ignorée", "80 arbres maximum"],
    answer: 1,
    explanation: "Comme le bootstrap en Random Forest, mais pour le boosting. Chaque arbre ne voit que 80% des lignes → régularisation supplémentaire, réduit l'overfitting." },

  { id: 90, category: "Boosting", difficulty: "moyen",
    question: "La val_loss remonte après l'itération 80 mais le train_loss continue de baisser. Que se passe-t-il ?",
    choices: ["Le modèle converge", "Overfitting — le modèle mémorise le train sans généraliser", "Underfitting", "Bug dans le code"],
    answer: 1,
    explanation: "Train qui baisse + val qui remonte = le modèle commence à mémoriser le bruit. C'est exactement ce que l'early stopping détecte pour arrêter l'entraînement." },

  { id: 91, category: "Boosting", difficulty: "moyen",
    question: "Quelle est la différence entre <strong>XGBoost</strong> et <strong>LightGBM</strong> ?",
    choices: ["Aucune", "LightGBM est généralement plus rapide grâce au growth leaf-wise au lieu de level-wise", "XGBoost est toujours meilleur", "LightGBM est pour le deep learning"],
    answer: 1,
    explanation: "XGBoost fait grandir l'arbre niveau par niveau. LightGBM fait grandir feuille par feuille (la feuille avec le plus de gain en premier). Résultat : LightGBM est souvent 2-5× plus rapide sur de gros datasets." },

  { id: 92, category: "Boosting", difficulty: "facile",
    question: "Quel est le principal avantage de XGBoost sur la régression linéaire ?",
    choices: ["Il est plus rapide", "Il capte les relations non-linéaires et les interactions entre features", "Il est plus interprétable", "Il n'a pas besoin de données"],
    answer: 1,
    explanation: "La régression linéaire suppose y = somme pondérée des features. XGBoost peut apprendre que 'surface > 80 ET ville = Paris → +50K' sans qu'on crée la feature manuellement." },

  { id: 93, category: "Boosting", difficulty: "difficile",
    question: "XGBoost gère nativement les valeurs manquantes. Comment ?",
    choices: ["Il les supprime", "Il apprend la meilleure direction (gauche ou droite) pour les NaN à chaque split", "Il les remplace par 0", "Il les ignore"],
    answer: 1,
    explanation: "À chaque split, XGBoost teste les deux directions pour les NaN et choisit celle qui réduit le plus la loss. C'est un avantage par rapport à sklearn qui exige un imputer." },

  { id: 94, category: "Boosting", difficulty: "moyen",
    question: "Pourquoi le boosting est-il plus sensible à l'overfitting que le Random Forest ?",
    choices: ["Il est plus lent", "Chaque arbre corrige le précédent → il peut finir par mémoriser le bruit des résidus", "Il utilise plus de mémoire", "Ce n'est pas vrai"],
    answer: 1,
    explanation: "Le boosting se concentre sur les erreurs résiduelles. Si ces erreurs sont du bruit (irréductible), le modèle les mémorise quand même. D'où l'importance de l'early stopping et de la régularisation." },

  # ── KNN & SVM (9 questions supplémentaires) ──

  { id: 95, category: "KNN & SVM", difficulty: "facile",
    question: "Pourquoi KNN est-il appelé 'lazy learner' ?",
    choices: ["Il est lent", "Il ne construit pas de modèle au .fit() — il stocke juste les données", "Il n'apprend que les features importantes", "Il oublie vite"],
    answer: 1,
    explanation: "Le .fit() de KNN ne calcule rien — il mémorise le dataset. Tout le travail est fait au .predict() quand il calcule les distances. C'est l'opposé d'un 'eager learner' comme la régression." },

  { id: 96, category: "KNN & SVM", difficulty: "moyen",
    question: "K=1 en KNN donne un train score de 100%. Pourquoi ?",
    choices: ["Le modèle est parfait", "Chaque point est son propre voisin le plus proche → il se prédit lui-même", "Bug", "C'est normal pour tout K"],
    answer: 1,
    explanation: "Avec K=1, le voisin le plus proche d'un point du train est lui-même (distance 0). Le modèle 'triche' en se rappelant chaque donnée. C'est l'overfitting maximal." },

  { id: 97, category: "KNN & SVM", difficulty: "moyen",
    question: "Comment choisir le bon K en KNN ?",
    choices: ["Toujours K=5", "Tester plusieurs K et prendre celui qui maximise le score sur le test/validation set", "K = nombre de features", "K = nombre de classes"],
    answer: 1,
    explanation: "On trace train score et test score en fonction de K. K petit → overfitting (train élevé, test bas). K grand → underfitting (les deux baissent). On cherche le K où le test score est maximal." },

  { id: 98, category: "KNN & SVM", difficulty: "difficile",
    question: "KNN avec 1 million de lignes et 50 features est très lent en prédiction. Pourquoi ?",
    choices: ["Le modèle est trop complexe", "À chaque prédiction, KNN calcule 1 million de distances en 50 dimensions", "Il faut plus de RAM", "Le scaling est trop long"],
    answer: 1,
    explanation: "KNN calcule la distance euclidienne entre le nouveau point et chaque point d'entraînement. 1M × 50 dimensions = énorme. Complexité en O(n×d) par prédiction. Solutions : KD-Tree, Ball-Tree, ou changer de modèle." },

  { id: 99, category: "KNN & SVM", difficulty: "moyen",
    question: "Qu'est-ce que le <strong>kernel trick</strong> en SVM ?",
    choices: ["Un raccourci de calcul", "Projeter implicitement les données dans un espace de dimension supérieure pour les rendre linéairement séparables", "Supprimer les outliers", "Normaliser les données"],
    answer: 1,
    explanation: "Si les données ne sont pas séparables par une droite en 2D, le kernel RBF les projette en dimension infinie où une frontière linéaire existe. Et grâce au 'trick', on n'a pas besoin de calculer explicitement la projection." },

  { id: 100, category: "KNN & SVM", difficulty: "facile",
    question: "Qu'est-ce que la <strong>marge</strong> en SVM ?",
    choices: ["L'erreur de prédiction", "La distance entre la frontière de décision et les points les plus proches de chaque classe", "Le nombre de support vectors", "La régularisation"],
    answer: 1,
    explanation: "SVM maximise la marge = distance entre l'hyperplan et les points les plus proches (support vectors). Plus la marge est grande, meilleure est la généralisation." },

  { id: 101, category: "KNN & SVM", difficulty: "moyen",
    question: "Le paramètre <code>C</code> en SVM contrôle quoi ?",
    choices: ["Le nombre de clusters", "Le compromis entre marge large (généralisation) et classification correcte des points d'entraînement", "Le kernel", "La vitesse"],
    answer: 1,
    explanation: "C petit → marge large, accepte des erreurs → meilleure généralisation. C grand → marge étroite, essaie de classer tous les points correctement → risque overfitting." },

  { id: 102, category: "KNN & SVM", difficulty: "difficile",
    question: "Pourquoi SVM ne scale-t-il pas bien au-delà de ~10 000 lignes ?",
    choices: ["Il n'est pas implémenté pour ça", "La complexité d'entraînement est entre O(n²) et O(n³) → quadratique à cubique", "Il a trop de paramètres", "Il a besoin d'un GPU"],
    answer: 1,
    explanation: "SVM résout un problème d'optimisation quadratique dont la complexité est entre O(n²) et O(n³). Avec 100K lignes, ça devient impraticable. Pour les gros datasets, préférer XGBoost ou LinearSVC." },

  { id: 103, category: "KNN & SVM", difficulty: "moyen",
    question: "Que sont les <strong>support vectors</strong> ?",
    choices: ["Tous les points du dataset", "Les points les plus proches de la frontière de décision — ceux qui 'supportent' l'hyperplan", "Les outliers supprimés", "Les features sélectionnées"],
    answer: 1,
    explanation: "Seuls les points proches de la frontière (support vectors) influencent la position de l'hyperplan. Les autres points pourraient être supprimés sans changer le modèle. C'est ce qui rend SVM efficace en haute dimension." },
)
