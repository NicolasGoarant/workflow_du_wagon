# frozen_string_literal: true

QUIZ_QUESTIONS = [
  # ── PREPROCESSING (8 questions) ──
  { id: 1, category: "Preprocessing", difficulty: "facile",
    question: "Que renvoie <code>df.shape</code> ?",
    choices: ["Le nombre de colonnes uniquement", "Un tuple (lignes, colonnes)", "La taille en mémoire du DataFrame", "Les types de chaque colonne"],
    answer: 1,
    explanation: "<code>df.shape</code> renvoie un tuple <code>(n_rows, n_cols)</code>. C'est le premier réflexe pour connaître la taille du dataset." },

  { id: 2, category: "Preprocessing", difficulty: "moyen",
    question: "Pourquoi utilise-t-on la <strong>médiane</strong> plutôt que la moyenne pour imputer les valeurs manquantes numériques ?",
    choices: ["La médiane est plus rapide à calculer", "La médiane est robuste aux outliers", "La médiane remplit plus de valeurs", "La moyenne n'est pas supportée par sklearn"],
    answer: 1,
    explanation: "Si tu as des surfaces de 18 à 450 m², la moyenne sera tirée vers le haut par les outliers. La médiane, elle, reste au centre réel de la distribution." },

  { id: 3, category: "Preprocessing", difficulty: "difficile",
    question: "Quel est le risque de faire <code>scaler.fit_transform(X)</code> sur le dataset COMPLET avant le train/test split ?",
    choices: ["Le modèle sera plus lent", "Data leakage : le scaler a vu les données de test", "Les colonnes catégorielles vont crasher", "Aucun risque, c'est une bonne pratique"],
    answer: 1,
    explanation: "C'est la source #1 de data leakage. Le scaler calcule μ et σ en incluant le test set → le modèle a indirectement 'vu' les données de test. Toujours : split → fit(train) → transform(test)." },

  { id: 4, category: "Preprocessing", difficulty: "moyen",
    question: "Quand utilise-t-on le <strong>One-Hot Encoding</strong> plutôt que l'<strong>Ordinal Encoding</strong> ?",
    choices: ["Quand il y a beaucoup de catégories", "Quand les catégories n'ont PAS d'ordre naturel", "Quand les catégories sont numériques", "Quand on utilise un arbre de décision"],
    answer: 1,
    explanation: "Paris, Lyon, Marseille → pas d'ordre. Encoder 1, 2, 3 impliquerait que Marseille > Lyon > Paris. Le One-Hot crée une colonne binaire par catégorie, sans imposer d'ordre." },

  { id: 5, category: "Preprocessing", difficulty: "facile",
    question: "Quels modèles n'ont <strong>PAS besoin</strong> de scaling ?",
    choices: ["KNN et SVM", "Régression Linéaire et Logistique", "Arbres de décision et Random Forest", "Réseaux de neurones"],
    answer: 2,
    explanation: "Les arbres de décision posent des questions (surface > 80 ?). L'échelle absolue n'a pas d'importance — seul l'ordre compte. KNN, SVM, régressions et réseaux de neurones, eux, sont sensibles à l'échelle." },

  { id: 6, category: "Preprocessing", difficulty: "moyen",
    question: "Que fait <code>df[\"surface\"].clip(lower, upper)</code> ?",
    choices: ["Supprime les lignes hors bornes", "Remplace les valeurs hors bornes par les bornes (capping)", "Crée une colonne binaire is_outlier", "Normalise entre 0 et 1"],
    answer: 1,
    explanation: "Le clipping (capping) remplace 450 m² par la borne supérieure (ex: 238 m²). Moins de perte de données qu'une suppression, mais attention à ne pas déformer la distribution." },

  { id: 7, category: "Preprocessing", difficulty: "facile",
    question: "Quel est le ratio train/test le plus courant ?",
    choices: ["50/50", "70/30 ou 80/20", "90/10", "60/40"],
    answer: 1,
    explanation: "70/30 est le standard. Pour un petit dataset (< 1000 lignes), on peut monter à 80/20 pour donner plus de données au modèle." },

  { id: 8, category: "Preprocessing", difficulty: "difficile",
    question: "Un code postal (75001, 69003, 13001) est stocké en <code>int64</code>. Faut-il le traiter comme numérique ou catégoriel ?",
    choices: ["Numérique — c'est un nombre", "Catégoriel — le nombre n'a pas de sens mathématique", "Ça dépend du modèle", "Il faut le supprimer"],
    answer: 1,
    explanation: "75001 + 69003 = 144004 n'a aucun sens. Un code postal est un identifiant, pas une quantité. Il faut le convertir en object/string puis l'encoder en catégoriel." },

  # ── EDA (4 questions) ──
  { id: 9, category: "EDA", difficulty: "facile",
    question: "Que mesure le coefficient de corrélation de Pearson ?",
    choices: ["La causalité entre deux variables", "La relation LINÉAIRE entre deux variables", "La distance entre deux distributions", "Le pourcentage de valeurs manquantes communes"],
    answer: 1,
    explanation: "Corrélation ≠ causalité ! Le coefficient mesure uniquement la force de la relation linéaire, de -1 (inversement proportionnel) à +1 (proportionnel). 0 = pas de relation linéaire (mais peut-être non-linéaire)." },

  { id: 10, category: "EDA", difficulty: "moyen",
    question: "Deux features ont une corrélation de 0.85 entre elles. Quel problème cela pose-t-il en régression linéaire ?",
    choices: ["Le modèle sera plus lent", "Multicolinéarité : les coefficients deviennent instables", "Les prédictions seront toujours fausses", "Aucun problème"],
    answer: 1,
    explanation: "Quand deux variables disent presque la même chose, la régression ne sait pas à laquelle attribuer l'effet. Les coefficients oscillent entre des valeurs extrêmes. Solution : VIF, Ridge/Lasso, ou supprimer une des deux." },

  { id: 11, category: "EDA", difficulty: "moyen",
    question: "Pourquoi appliquer <code>np.log1p()</code> sur la variable target (prix) ?",
    choices: ["Pour accélérer le calcul", "Pour rendre une distribution skewed plus gaussienne", "Pour supprimer les outliers", "Pour normaliser entre 0 et 1"],
    answer: 1,
    explanation: "Les prix immobiliers ont une longue traîne à droite (quelques villas très chères). Le log compresse les grandes valeurs et étire les petites → distribution plus symétrique, mieux adaptée à beaucoup de modèles." },

  { id: 12, category: "EDA", difficulty: "facile",
    question: "Quelle commande Seaborn affiche TOUS les scatter plots 2 à 2 en un coup ?",
    choices: ["sns.heatmap()", "sns.pairplot()", "sns.boxplot()", "sns.scatterplot()"],
    answer: 1,
    explanation: "pairplot() crée une matrice de scatter plots entre chaque paire de variables numériques. Avec <code>corner=True</code>, on évite la redondance (moitié inférieure seulement)." },

  # ── RÉGRESSION LINÉAIRE (4 questions) ──
  { id: 13, category: "Régression Linéaire", difficulty: "facile",
    question: "Que mesure le <strong>R²</strong> (coefficient de détermination) ?",
    choices: ["L'erreur en euros", "Le % de variance expliquée par le modèle", "Le nombre de features utiles", "La corrélation entre features"],
    answer: 1,
    explanation: "R² = 0.85 signifie que le modèle capte 85% de la variabilité du prix. Les 15% restants sont du bruit ou des facteurs non captés. 1.0 = parfait (suspect), 0 = le modèle ne fait pas mieux que la moyenne." },

  { id: 14, category: "Régression Linéaire", difficulty: "moyen",
    question: "Quelle est la différence fondamentale entre Ridge (L2) et Lasso (L1) ?",
    choices: ["Ridge est plus rapide", "Lasso peut mettre des coefficients à zéro (sélection de features)", "Ridge utilise la médiane au lieu de la moyenne", "Il n'y a aucune différence"],
    answer: 1,
    explanation: "Ridge réduit tous les coefficients mais n'en annule jamais. Lasso peut mettre des coefficients exactement à 0 → sélection automatique de features. Utile quand tu as beaucoup de features peu pertinentes." },

  { id: 15, category: "Régression Linéaire", difficulty: "moyen",
    question: "Le RMSE vaut 25 000. Que signifie cette valeur concrètement ?",
    choices: ["Le modèle se trompe de 25 000 en moyenne (dans l'unité de la target)", "Le modèle a 25 000 paramètres", "25 000 lignes sont mal prédites", "Le R² est de 0.25"],
    answer: 0,
    explanation: "Le RMSE est en unité de la target. Si tu prédis des prix en €, RMSE = 25 000€ signifie que le modèle se trompe en moyenne de ~25K€. Il pénalise fortement les grosses erreurs (contrairement au MAE)." },

  { id: 16, category: "Régression Linéaire", difficulty: "difficile",
    question: "Ton modèle a un R² de 0.95 en train et 0.62 en test. Que se passe-t-il ?",
    choices: ["Underfitting", "Overfitting", "Le modèle est parfait", "Data leakage"],
    answer: 1,
    explanation: "Train >> Test = le modèle a mémorisé le train sans généraliser. Solutions : régularisation (Ridge/Lasso), réduire la complexité (max_depth), plus de données, cross-validation." },

  # ── RÉGRESSION LOGISTIQUE (4 questions) ──
  { id: 17, category: "Régression Logistique", difficulty: "facile",
    question: "La régression logistique sort une valeur entre 0 et 1. Qu'est-ce que c'est ?",
    choices: ["Un score de corrélation", "Une probabilité (via la sigmoïde)", "Un coefficient de régression", "Le recall"],
    answer: 1,
    explanation: "La sigmoïde transforme la somme pondérée en probabilité. Si P(spam) = 0.87, le modèle est sûr à 87% que c'est du spam. Le seuil par défaut (0.5) détermine la classe finale." },

  { id: 18, category: "Régression Logistique", difficulty: "moyen",
    question: "Ton dataset a 95% de clients non-fraudeurs et 5% de fraudeurs. L'accuracy est de 95%. Le modèle est-il bon ?",
    choices: ["Oui, 95% c'est excellent", "Non — il peut prédire 'non-fraudeur' à chaque fois et avoir 95%", "On ne peut pas savoir", "Il faut regarder le R²"],
    answer: 1,
    explanation: "C'est le piège de l'accuracy sur classes déséquilibrées. Un modèle 'stupide' qui prédit toujours la classe majoritaire atteint 95%. Il faut regarder recall (détection), precision, et F1-score." },

  { id: 19, category: "Régression Logistique", difficulty: "moyen",
    question: "Tu veux détecter des fraudes bancaires. Quelle métrique privilégier ?",
    choices: ["Precision (minimiser les faux positifs)", "Recall (minimiser les faux négatifs)", "Accuracy", "R²"],
    answer: 1,
    explanation: "Un faux négatif = une fraude non détectée → perte financière. Mieux vaut bloquer quelques transactions légitimes (faux positifs) que laisser passer des fraudes. Le recall mesure 'parmi les vraies fraudes, combien sont détectées ?'." },

  { id: 20, category: "Régression Logistique", difficulty: "difficile",
    question: "L'AUC-ROC de ton modèle est de 0.52. Qu'est-ce que cela signifie ?",
    choices: ["Le modèle est excellent", "Le modèle ne fait guère mieux que le hasard", "Le modèle prédit l'inverse", "Le seuil est mal réglé"],
    answer: 1,
    explanation: "AUC = 0.50 = hasard pur (pile ou face). 0.52 est à peine mieux. Le modèle n'a pas appris de pattern discriminant. Il faut revoir les features, le preprocessing, ou changer de modèle." },

  # ── ARBRES & RANDOM FOREST (3 questions) ──
  { id: 21, category: "Arbres & Random Forest", difficulty: "moyen",
    question: "Pourquoi un Random Forest est-il meilleur qu'un seul arbre de décision ?",
    choices: ["Il est plus rapide", "Il moyenne N arbres aléatoires → réduit la variance (bagging)", "Il utilise le gradient", "Il fait de la sélection de features automatique"],
    answer: 1,
    explanation: "Un arbre seul overfitte facilement. Le Random Forest entraîne N arbres sur des sous-échantillons aléatoires (bootstrap) et fait voter la majorité. La moyenne de N opinions imparfaites est souvent meilleure qu'une seule." },

  { id: 22, category: "Arbres & Random Forest", difficulty: "facile",
    question: "Les arbres de décision ont-ils besoin de scaling ?",
    choices: ["Oui, toujours", "Non, jamais", "Seulement pour les features catégorielles", "Seulement si les features sont corrélées"],
    answer: 1,
    explanation: "Un arbre pose des questions du type 'surface > 80 ?'. Que la surface soit en m² ou en cm², le split sera au même endroit. Seul l'ordre des valeurs compte, pas l'échelle." },

  { id: 23, category: "Arbres & Random Forest", difficulty: "moyen",
    question: "Que mesure <code>model.feature_importances_</code> dans un Random Forest ?",
    choices: ["La corrélation avec la target", "La contribution de chaque feature aux splits de l'ensemble des arbres", "Le coefficient comme en régression linéaire", "Le nombre de NaN par feature"],
    answer: 1,
    explanation: "Chaque fois qu'un arbre utilise une feature pour un split, il mesure la réduction d'impureté. On somme sur tous les arbres et normalise. ⚠️ Attention : des features corrélées se partagent l'importance." },

  # ── BOOSTING / XGBOOST (3 questions) ──
  { id: 24, category: "Boosting", difficulty: "moyen",
    question: "Quelle est la différence fondamentale entre le <strong>bagging</strong> (Random Forest) et le <strong>boosting</strong> (XGBoost) ?",
    choices: ["Bagging utilise des arbres, boosting des réseaux de neurones", "Bagging entraîne en parallèle, boosting en séquentiel (chaque arbre corrige les erreurs du précédent)", "Boosting est toujours meilleur", "Ils sont identiques"],
    answer: 1,
    explanation: "Bagging : N arbres indépendants en parallèle → réduit la variance. Boosting : arbres séquentiels, chacun se concentre sur les erreurs du précédent → réduit le biais. Le boosting est souvent plus performant mais plus sensible à l'overfitting." },

  { id: 25, category: "Boosting", difficulty: "moyen",
    question: "À quoi sert l'<strong>early stopping</strong> dans XGBoost ?",
    choices: ["Accélérer le calcul", "Arrêter l'entraînement quand la performance sur le validation set stagne/décroît", "Supprimer les features inutiles", "Baisser le learning rate automatiquement"],
    answer: 1,
    explanation: "Sans early stopping, le boosting continue d'ajouter des arbres même quand il n'y a plus de gain → overfitting. L'early stopping surveille la val_loss et arrête quand elle remonte pendant N itérations (patience)." },

  { id: 26, category: "Boosting", difficulty: "difficile",
    question: "Si tu baisses le <code>learning_rate</code> de 0.3 à 0.05, que dois-tu faire en parallèle ?",
    choices: ["Baisser n_estimators", "Augmenter n_estimators", "Rien, c'est indépendant", "Changer le kernel"],
    answer: 1,
    explanation: "Un petit learning_rate fait des corrections plus prudentes → il faut plus d'arbres pour atteindre le même résultat. La règle : lr bas × n_estimators élevé = souvent meilleur, mais plus lent." },

  # ── KNN & SVM (3 questions) ──
  { id: 27, category: "KNN & SVM", difficulty: "facile",
    question: "Que fait KNN pour prédire la classe d'un nouveau point ?",
    choices: ["Il calcule une équation", "Il regarde les K voisins les plus proches et vote la majorité", "Il construit un arbre", "Il utilise le gradient"],
    answer: 1,
    explanation: "KNN est un 'lazy learner' : pas de vrai modèle appris. Pour chaque prédiction, il calcule la distance avec tous les points d'entraînement, prend les K plus proches, et vote. D'où l'importance du scaling." },

  { id: 28, category: "KNN & SVM", difficulty: "moyen",
    question: "Pourquoi le scaling est-il <strong>obligatoire</strong> pour KNN ?",
    choices: ["Pour accélérer le calcul", "Parce que KNN utilise des distances — sans scaling, la feature avec la plus grande échelle domine", "Pour réduire les NaN", "Ce n'est pas obligatoire"],
    answer: 1,
    explanation: "Si surface est en [18, 450] et rooms en [1, 12], la distance sera dominée par surface. Un écart de 100 m² comptera 10× plus qu'un écart de 10 pièces. Le scaling met tout sur la même échelle." },

  { id: 29, category: "KNN & SVM", difficulty: "moyen",
    question: "Que fait le kernel RBF dans un SVM ?",
    choices: ["Il réduit les dimensions", "Il projette les données dans un espace de dimension supérieure pour les rendre linéairement séparables", "Il supprime les outliers", "Il normalise les features"],
    answer: 1,
    explanation: "Si les classes ne sont pas séparables par une droite dans l'espace original, le kernel RBF les projette dans un espace plus grand où une frontière linéaire existe. C'est le 'truc' mathématique du SVM." },

  # ── K-MEANS & PCA (4 questions) ──
  { id: 30, category: "Non-supervisé", difficulty: "facile",
    question: "K-Means est un algorithme supervisé ou non-supervisé ?",
    choices: ["Supervisé — il a besoin de labels", "Non-supervisé — il trouve des groupes sans labels", "Semi-supervisé", "Ça dépend du dataset"],
    answer: 1,
    explanation: "K-Means n'a pas de target (y). Il cherche des regroupements naturels dans les données. C'est au Data Scientist d'interpréter les clusters après coup ('ce groupe = jeunes actifs')." },

  { id: 31, category: "Non-supervisé", difficulty: "moyen",
    question: "Comment choisir le bon K (nombre de clusters) en K-Means ?",
    choices: ["Toujours K=3", "Méthode du coude (inertia) + silhouette score", "On teste tous les K de 1 à 100", "Le modèle le choisit automatiquement"],
    answer: 1,
    explanation: "L'inertia diminue toujours avec K — on cherche le 'coude' où le gain marginal diminue brutalement. Le silhouette score (0 à 1) confirme en mesurant la qualité de séparation. Les deux ensemble donnent le bon K." },

  { id: 32, category: "Non-supervisé", difficulty: "moyen",
    question: "La PCA avec <code>n_components=0.95</code> fait quoi ?",
    choices: ["Garde 95 composantes", "Garde le nombre minimum de composantes qui captent 95% de la variance", "Réduit la variance de 95%", "Garde les features avec corrélation > 0.95"],
    answer: 1,
    explanation: "Au lieu de fixer un nombre, on fixe un seuil de variance. Si 5 composantes captent 95.6% de l'info sur 18 features, PCA garde 5 dimensions. On perd 4.4% d'info mais on divise les dimensions par ~4." },

  { id: 33, category: "Non-supervisé", difficulty: "facile",
    question: "Faut-il scaler les données AVANT d'appliquer PCA ?",
    choices: ["Non, PCA s'en occupe", "Oui, sinon la feature avec la plus grande variance domine les composantes", "Seulement si on a des NaN", "Seulement pour les catégorielles"],
    answer: 1,
    explanation: "Si 'revenu' est en [20K, 200K] et 'âge' en [18, 80], PC1 sera presque entièrement 'revenu'. Le scaling met tout sur la même échelle → PCA trouve les vrais axes de variance, pas les axes d'échelle." },

  # ── DEEP LEARNING (5 questions) ──
  { id: 34, category: "Deep Learning", difficulty: "facile",
    question: "Quelle fonction d'activation est le choix par défaut pour les couches cachées d'un réseau de neurones ?",
    choices: ["Sigmoid", "Softmax", "ReLU", "Tanh"],
    answer: 2,
    explanation: "ReLU = max(0, x). Simple, efficace, résout le problème du vanishing gradient. Sigmoid et tanh sont réservés à des cas spécifiques (dernière couche binaire pour sigmoid, RNN pour tanh)." },

  { id: 35, category: "Deep Learning", difficulty: "moyen",
    question: "Que fait le <strong>Dropout(0.3)</strong> ?",
    choices: ["Supprime 30% des données d'entraînement", "Éteint aléatoirement 30% des neurones à chaque batch d'entraînement", "Réduit le learning rate de 30%", "Garde seulement les 30% meilleurs neurones"],
    answer: 1,
    explanation: "À chaque forward pass, 30% des neurones sont mis à 0 aléatoirement. Le réseau ne peut pas compter sur un seul chemin → force la redondance → régularisation naturelle contre l'overfitting." },

  { id: 36, category: "Deep Learning", difficulty: "moyen",
    question: "Pour de la classification binaire (spam/pas spam), quelle combinaison activation + loss ?",
    choices: ["ReLU + MSE", "Sigmoid + binary_crossentropy", "Softmax + categorical_crossentropy", "Tanh + MAE"],
    answer: 1,
    explanation: "Sigmoid sort une probabilité [0,1] → parfait pour binaire. Binary crossentropy mesure la différence entre la proba prédite et le label (0 ou 1). C'est LE combo pour la classification binaire." },

  { id: 37, category: "Deep Learning", difficulty: "difficile",
    question: "Qu'est-ce que le <strong>Transfer Learning</strong> avec VGG16 ?",
    choices: ["Entraîner VGG16 de zéro sur nos données", "Réutiliser les couches convolutives pré-entraînées sur ImageNet et ne ré-entraîner que les dernières couches", "Copier les prédictions d'un autre modèle", "Transférer les données d'un dataset à un autre"],
    answer: 1,
    explanation: "VGG16 a appris à détecter des bords, textures, formes sur 14M d'images. On garde ces 'yeux' (base.trainable=False) et on ré-entraîne juste la dernière couche pour notre tâche. Efficace même avec peu de données." },

  { id: 38, category: "Deep Learning", difficulty: "moyen",
    question: "Pourquoi utilise-t-on un LSTM plutôt qu'un RNN classique ?",
    choices: ["Le LSTM est plus rapide", "Le LSTM a des portes (forget, input, output) qui résolvent le vanishing gradient sur les longues séquences", "Le LSTM utilise des convolutions", "Il n'y a aucune différence"],
    answer: 1,
    explanation: "Un RNN classique 'oublie' les informations lointaines car le gradient disparaît au fil des couches. Le LSTM a un mécanisme de portes qui décide quoi oublier et quoi retenir → mémoire à long terme." },

  # ── MÉTHODES ML (4 questions) ──
  { id: 39, category: "Méthodes ML", difficulty: "facile",
    question: "Quelle est la bonne séquence pour le preprocessing + prédiction ?",
    choices: ["fit(test) → transform(train) → predict(test)", "fit_transform(train) → transform(test) → predict(test)", "transform(all) → fit(train) → predict(test)", "predict(test) → fit(train) → transform(test)"],
    answer: 1,
    explanation: "fit_transform sur le train (apprend et applique), puis transform sur le test (applique sans ré-apprendre), puis predict. C'est LA séquence à retenir." },

  { id: 40, category: "Méthodes ML", difficulty: "moyen",
    question: "Quelle est la différence entre <code>.score()</code> (sklearn) et <code>.evaluate()</code> (Keras) ?",
    choices: ["Aucune différence", ".score() renvoie accuracy ou R², .evaluate() renvoie la loss ET les métriques", ".evaluate() est plus rapide", ".score() est pour Keras, .evaluate() pour sklearn"],
    answer: 1,
    explanation: ".score() est un raccourci sklearn qui renvoie la métrique par défaut (accuracy en classif, R² en régression). .evaluate() est Keras et renvoie un tuple (loss, metric1, metric2...) — plus riche." },
]
