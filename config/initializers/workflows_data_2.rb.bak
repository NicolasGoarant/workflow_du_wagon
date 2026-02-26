# frozen_string_literal: true
# Workflows 3-15: Supervised, Unsupervised, DL, MLOps

WORKFLOWS.push(
  # ============================================================
  # LINEAR REGRESSION
  # ============================================================
  {
    slug: "linreg", title: "Régression Linéaire", icon: "📈",
    badge: "Supervisé", badge_color: "accent",
    subtitle: "Prédire une valeur continue — y = β₀ + β₁x₁ + ... + βₙxₙ",
    analogy_title: "L'analogie du fil tendu",
    analogy_text: "Imagine un nuage de points (surface vs prix). La régression linéaire trouve le fil tendu qui passe au plus près de tous les points. Ce fil te permet de prédire : 'pour 80 m², le prix devrait être autour de X'.",
    steps: [
      { title: "Vérifier les hypothèses", explain: "La régression a des conditions d'utilisation. Si la relation X-y est une courbe, le modèle sera mauvais.",
        code_block: <<~PY,
          from statsmodels.stats.outliers_influence import variance_inflation_factor

          # Multicolinéarité : VIF (Variance Inflation Factor)
          X_num = df[["surface", "rooms", "age"]]
          for i, col in enumerate(X_num.columns):
              vif = variance_inflation_factor(X_num.values, i)
              print(f"VIF {col:10} : {vif:.1f}")
        PY
        output: <<~OUT,
          VIF surface    :  2.8   ✅ (< 5)
          VIF rooms      :  2.4   ✅ (< 5)
          VIF age        :  1.1   ✅ (< 5)

          → Si VIF > 5, supprimer la feature ou utiliser Ridge.
        OUT
        code_notes: [
          { marker: "VIF", text: "Mesure à quel point une feature est expliquée par les autres. VIF > 5 = multicolinéarité problématique. VIF > 10 = critique." },
        ] },
      { title: "Entraîner et interpréter les coefficients", explain: "Le modèle cherche la droite minimisant la somme des erreurs au carré. Après scaling, les coefficients sont comparables.",
        code_block: <<~PY,
          from sklearn.linear_model import LinearRegression
          from sklearn.preprocessing import StandardScaler
          import numpy as np

          scaler = StandardScaler()
          X_train_sc = scaler.fit_transform(X_train)
          X_test_sc  = scaler.transform(X_test)

          model = LinearRegression()
          model.fit(X_train_sc, y_train)

          # Coefficients (importances relatives)
          for feat, coef in zip(feature_names, model.coef_):
              print(f"  {feat:15} : {coef:+.0f} €")
          print(f"  {'Intercept':15} : {model.intercept_:+.0f} €")
        PY
        output: <<~OUT,
          surface         : +2850 €  ← +1 std de surface = +2850€
          rooms           : +1200 €
          age             :  -980 €  ← plus c'est vieux, moins cher
          city_Paris      : +4500 €  ← prime Paris
          Intercept       : +245800 €
        OUT
        code_notes: [
          { marker: "coef_", text: "Après scaling, les coefficients sont COMPARABLES. <code>surface: +2850</code> = augmenter surface d'1 std augmente le prix de 2850 €." },
          { marker: "intercept_", text: "Le prix quand toutes les features sont à leur moyenne (grâce au scaling). Ici ~245K€." },
        ] },
      { title: "Régulariser si overfitting", explain: "Ridge réduit tous les coefficients, Lasso en met certains à zéro (sélection de features automatique).",
        code_block: <<~PY,
          from sklearn.linear_model import Ridge, Lasso

          ridge = Ridge(alpha=1.0).fit(X_train_sc, y_train)
          lasso = Lasso(alpha=100).fit(X_train_sc, y_train)

          print(f"{'Feature':15} {'OLS':>8} {'Ridge':>8} {'Lasso':>8}")
          for i, feat in enumerate(feature_names):
              print(f"{feat:15} {model.coef_[i]:+8.0f} {ridge.coef_[i]:+8.0f} {lasso.coef_[i]:+8.0f}")
        PY
        output: <<~OUT,
          Feature             OLS    Ridge    Lasso
          surface           +2850    +2720    +2600
          rooms             +1200    +1150        0  ← Lasso l'a éliminé !
          age                -980     -950     -890
          city_Paris        +4500    +4200    +3800
          city_Lyon          +800     +750        0  ← éliminé aussi
        OUT
        code_notes: [
          { marker: "Ridge(alpha=1.0)", text: "Pénalité L2 : réduit TOUS les coefficients mais aucun à zéro. Bon si toutes les features comptent." },
          { marker: "Lasso(alpha=100)", text: "Pénalité L1 : met les features peu utiles à exactement 0. C'est de la sélection de features automatique." },
        ] },
      { title: "Évaluer et cross-valider", explain: "Mesurer la performance et vérifier la stabilité.",
        code_block: <<~PY,
          from sklearn.model_selection import cross_val_score
          from sklearn.metrics import mean_squared_error, r2_score

          y_pred = model.predict(X_test_sc)
          print(f"R² train : {model.score(X_train_sc, y_train):.3f}")
          print(f"R² test  : {r2_score(y_test, y_pred):.3f}")
          print(f"RMSE     : {np.sqrt(mean_squared_error(y_test, y_pred)):,.0f} €")

          cv = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="r2")
          print(f"\\nCV R² : {cv.mean():.3f} ± {cv.std():.3f}")
        PY
        output: <<~OUT,
          R² train : 0.842
          R² test  : 0.831  ← proche du train → pas d'overfitting
          RMSE     : 54,200 €

          CV R² : 0.835 ± 0.018  ← stable (faible std)
        OUT
        code_notes: [
          { marker: "R²", text: "% de variance expliquée. 0.83 = le modèle explique 83% des variations de prix." },
          { marker: "RMSE", text: "Erreur en euros. Le modèle se trompe de ~54K€ en moyenne." },
          { marker: "CV ± 0.018", text: "Faible variation entre folds → modèle stable. Si std > 0.05, le modèle est instable." },
        ] },
    ],
    tips: { title: "📏 Métriques régression", items: [
      "R² : % de variance expliquée. 0.85 = capte 85% du signal",
      "RMSE : erreur en unité de la target (€). Interprétable directement",
      "MAE : erreur absolue moyenne, plus robuste aux outliers",
      "R² train >> R² test → overfitting → régulariser",
    ]},
    code_filename: "linear_regression.py",
    code_content: <<~PY,
      from sklearn.linear_model import LinearRegression, Ridge
      from sklearn.model_selection import train_test_split, cross_val_score
      from sklearn.preprocessing import StandardScaler
      from sklearn.metrics import mean_squared_error, r2_score
      import numpy as np

      scaler = StandardScaler()
      X_train_sc = scaler.fit_transform(X_train)
      X_test_sc  = scaler.transform(X_test)

      model = LinearRegression().fit(X_train_sc, y_train)
      y_pred = model.predict(X_test_sc)
      print(f"R²={r2_score(y_test,y_pred):.3f}, RMSE={np.sqrt(mean_squared_error(y_test,y_pred)):,.0f}€")

      cv = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="r2")
      print(f"CV R²: {cv.mean():.3f} ± {cv.std():.3f}")
    PY
  },

  # ============================================================
  # LOGISTIC REGRESSION
  # ============================================================
  {
    slug: "logreg", title: "Régression Logistique", icon: "🎯",
    badge: "Classification", badge_color: "accent",
    subtitle: "Prédire une catégorie (oui/non, spam/pas spam) via une probabilité",
    analogy_title: "L'analogie du thermomètre de confiance",
    analogy_text: "La régression logistique prend les symptômes et sort une probabilité entre 0 et 100 %. Au-delà du seuil (50 % par défaut), elle dit 'malade'. Mais tu peux ajuster : si rater un malade est grave, baisse le seuil à 30 %.",
    steps: [
      { title: "Vérifier l'équilibre des classes", explain: "Si 95% des emails sont 'pas spam', un modèle qui dit toujours 'pas spam' a 95% d'accuracy. Trompeur !",
        code_block: <<~PY,
          print(y_train.value_counts())
          print(f"\\nRatio : {y_train.value_counts(normalize=True).round(3).to_dict()}")
          # Si déséquilibré → class_weight='balanced'
        PY
        output: <<~OUT,
          0    892  (pas spam)
          1    118  (spam)
          Ratio : {0: 0.883, 1: 0.117}
          ⚠️ 12% de spam → class_weight='balanced'
        OUT
        code_notes: [
          { marker: "class_weight='balanced'", text: "Donne un poids inversement proportionnel à la fréquence. Les 118 spams pèseront ~7.5× plus dans la loss." },
        ] },
      { title: "Entraîner et évaluer", explain: "La sigmoïde est sensible à l'échelle → scaling obligatoire. Le paramètre C contrôle la régularisation.",
        code_block: <<~PY,
          from sklearn.linear_model import LogisticRegression
          from sklearn.metrics import classification_report, confusion_matrix

          model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
          model.fit(X_train_sc, y_train)

          y_pred = model.predict(X_test_sc)
          y_proba = model.predict_proba(X_test_sc)[:, 1]

          print(confusion_matrix(y_test, y_pred))
          print(classification_report(y_test, y_pred, target_names=["Pas spam", "Spam"]))
        PY
        output: <<~OUT,
          [[365  17]     TN=365  FP=17 (mails légitimes bloqués)
           [  8  43]]    FN=8    TP=43 (spams détectés)

                         precision  recall  f1-score
          Pas spam          0.98     0.96     0.97
          Spam              0.72     0.84     0.78
          accuracy                            0.94
        OUT
        code_notes: [
          { marker: "C=1.0", text: "C est l'INVERSE de la régularisation. C grand → colle aux données. C petit → plus simple." },
          { marker: "precision 0.72", text: "Parmi les prédits 'spam', 72% l'étaient. 28% de faux positifs (mails légitimes bloqués)." },
          { marker: "recall 0.84", text: "Parmi les vrais spams, 84% détectés. 16% sont passés à travers." },
        ] },
      { title: "Ajuster le seuil + courbe ROC", explain: "Seuil par défaut 0.5. Pour la fraude → le baisser. Pour le spam → le monter.",
        code_block: <<~PY,
          from sklearn.metrics import roc_auc_score, roc_curve

          fpr, tpr, thresholds = roc_curve(y_test, y_proba)
          auc = roc_auc_score(y_test, y_proba)
          print(f"AUC = {auc:.2f}")

          plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
          plt.plot([0, 1], [0, 1], "k--", label="Random (0.50)")
          plt.xlabel("Faux positifs"); plt.ylabel("Vrais positifs")
          plt.legend()

          # Seuil custom (ex: détection fraude → seuil bas)
          y_pred_03 = (y_proba >= 0.3).astype(int)
          print(classification_report(y_test, y_pred_03))
        PY
        output: <<~OUT,
          AUC = 0.94  ← excellent (1.0 = parfait, 0.5 = hasard)

          Seuil 0.3 vs 0.5 :
          Spam recall : 0.84 → 0.94  (+10% de détection)
          Spam precision : 0.72 → 0.58  (plus de faux positifs)
          → Choix métier : quel type d'erreur est le plus grave ?
        OUT
        code_notes: [
          { marker: "AUC = 0.94", text: "Résume la performance à TOUS les seuils. > 0.9 = excellent. < 0.7 = médiocre." },
          { marker: "seuil 0.3", text: "Le modèle est plus 'paranoïaque'. Recall monte mais precision baisse. C'est un compromis métier." },
        ] },
    ],
    tips: { title: "📏 Métriques classification", items: [
      "Accuracy : % bonnes prédictions — trompeuse si classes déséquilibrées",
      "Precision : parmi les prédits positifs, combien le sont vraiment ?",
      "Recall : parmi les vrais positifs, combien sont détectés ?",
      "F1-Score : moyenne harmonique precision/recall",
      "AUC-ROC : performance globale indépendante du seuil",
    ]},
    code_filename: "logistic_regression.py",
    code_content: <<~PY,
      from sklearn.linear_model import LogisticRegression
      from sklearn.metrics import classification_report, roc_auc_score

      model = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
      model.fit(X_train_sc, y_train)
      y_proba = model.predict_proba(X_test_sc)[:, 1]
      print(classification_report(y_test, model.predict(X_test_sc)))
      print(f"AUC: {roc_auc_score(y_test, y_proba):.3f}")
    PY
  },

  # ============================================================
  # DECISION TREES & RANDOM FOREST
  # ============================================================
  {
    slug: "trees", title: "Arbres & Random Forest", icon: "🌳",
    badge: "Supervisé", badge_color: "green",
    subtitle: "Prédire en posant des questions successives — comme 'Qui est-ce ?'",
    analogy_title: "L'analogie du jeu 'Qui est-ce ?'",
    analogy_text: "Un arbre pose la question qui élimine le plus de candidats. 'Surface > 80m² ? → Oui. Paris ? → Non.' Le Random Forest demande l'avis à 100 joueurs avec chacun un sous-ensemble différent, puis prend le vote majoritaire.",
    steps: [
      { title: "Decision Tree → Random Forest", explain: "Un arbre seul overfitte. 100 arbres diversifiés (bagging) sont bien plus fiables. PAS de scaling nécessaire.",
        code_block: <<~PY,
          from sklearn.ensemble import RandomForestClassifier
          from sklearn.tree import export_text

          # PAS de scaling ! L'arbre fait des comparaisons (>, ≤)
          rf = RandomForestClassifier(
              n_estimators=100,       # 100 arbres
              max_depth=10,           # limiter la profondeur
              max_features="sqrt",    # √n features par split
              min_samples_leaf=5,     # min 5 obs par feuille
              random_state=42, n_jobs=-1
          )
          rf.fit(X_train, y_train)

          print(f"Train : {rf.score(X_train, y_train):.3f}")
          print(f"Test  : {rf.score(X_test, y_test):.3f}")

          # Feature importance
          import pandas as pd
          feat_imp = pd.Series(rf.feature_importances_, index=feature_names)
          print(feat_imp.sort_values(ascending=False).head(5))
        PY
        output: <<~OUT,
          Train : 0.952
          Test  : 0.918  ← meilleur qu'un arbre seul (0.864)

          Feature importances :
          surface         0.342  ← la plus importante
          rooms           0.198
          age             0.156
          city_Paris      0.128
          price_per_m2    0.089
        OUT
        code_notes: [
          { marker: "n_estimators=100", text: "100 arbres votent. Plus = plus stable mais plus lent. 100-300 est un bon compromis." },
          { marker: "max_features='sqrt'", text: "Chaque split choisit parmi √n features. Force la diversité entre arbres." },
          { marker: "feature_importances_", text: "Réduction moyenne de l'impureté Gini apportée par chaque feature. ⚠️ Features corrélées se partagent l'importance." },
        ] },
      { title: "Tuning avec RandomizedSearchCV", explain: "Tester des combinaisons d'hyperparamètres aléatoires, puis affiner.",
        code_block: <<~PY,
          from sklearn.model_selection import RandomizedSearchCV

          params = {
              "n_estimators": [50, 100, 200, 300],
              "max_depth": [5, 10, 15, 20, None],
              "min_samples_leaf": [1, 3, 5, 10],
              "max_features": ["sqrt", "log2", 0.3]
          }

          search = RandomizedSearchCV(
              RandomForestClassifier(random_state=42),
              params, n_iter=30, cv=5, scoring="f1", n_jobs=-1
          )
          search.fit(X_train, y_train)
          print(f"Best params : {search.best_params_}")
          print(f"Best CV F1  : {search.best_score_:.3f}")
        PY
        output: <<~OUT,
          Best params : {n_estimators: 200, max_depth: 15,
                         min_samples_leaf: 3, max_features: 'sqrt'}
          Best CV F1  : 0.924
        OUT
        code_notes: [
          { marker: "n_iter=30", text: "30 combinaisons aléatoires sur 240 possibles. Plus rapide qu'exhaustif, souvent aussi bon." },
          { marker: "scoring='f1'", text: "Optimise le F1 plutôt que l'accuracy (classes déséquilibrées)." },
        ] },
    ],
    tips: { title: "🔑 Bagging vs Boosting", items: [
      "Bagging (Random Forest) : arbres indépendants en parallèle → réduit la variance",
      "Boosting (XGBoost) : arbres séquentiels corrigeant les erreurs → réduit le biais",
      "RF : robuste par défaut, moins de risque d'overfitting",
      "Boosting : souvent plus performant mais plus sensible aux hyperparamètres",
    ]},
    code_filename: "random_forest.py",
    code_content: <<~PY,
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.model_selection import RandomizedSearchCV

      rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
      rf.fit(X_train, y_train)
      print(f"Test: {rf.score(X_test, y_test):.3f}")

      params = {"n_estimators": [100,200], "max_depth": [5,10,None], "min_samples_leaf": [1,5,10]}
      search = RandomizedSearchCV(rf, params, n_iter=20, cv=5, scoring="f1", n_jobs=-1)
      search.fit(X_train, y_train)
      print(f"Best: {search.best_score_:.3f}")
    PY
  },

  # ============================================================
  # GRADIENT BOOSTING / XGBOOST
  # ============================================================
  {
    slug: "boosting", title: "Gradient Boosting / XGBoost", icon: "🚀",
    badge: "Supervisé", badge_color: "orange",
    subtitle: "Des arbres qui apprennent de leurs erreurs — le roi des données tabulaires",
    analogy_title: "L'analogie des couches de peinture",
    analogy_text: "Tu peins un portrait. La première couche est grossière. Chaque couche affine en corrigeant les erreurs de la précédente. 100 petits arbres 'moyens' qui se corrigent = souvent le meilleur modèle pour les données tabulaires.",
    steps: [
      { title: "Entraîner avec Early Stopping", explain: "Plutôt que deviner le bon nombre d'arbres, on surveille la performance sur un validation set et on arrête quand elle stagne. Comme arrêter de cuire un gâteau quand il est doré.",
        code_block: <<~PY,
          from xgboost import XGBClassifier
          from sklearn.model_selection import train_test_split

          # Validation set DANS le train (pour early stopping)
          X_tr, X_val, y_tr, y_val = train_test_split(
              X_train, y_train, test_size=0.2, random_state=42
          )

          xgb = XGBClassifier(
              n_estimators=500,            # max 500 arbres
              learning_rate=0.05,          # chaque arbre corrige doucement
              max_depth=6,                 # arbres peu profonds
              subsample=0.8,              # 80% des données par arbre
              colsample_bytree=0.8,       # 80% des features par arbre
              early_stopping_rounds=50,    # stop si pas d'amélioration ×50
              eval_metric="logloss",
              random_state=42
          )

          xgb.fit(
              X_tr, y_tr,
              eval_set=[(X_val, y_val)],   # surveiller la val_loss
              verbose=50
          )
          print(f"Arrêté à l'itération : {xgb.best_iteration}")
        PY
        output: <<~OUT,
          [0]   validation_0-logloss: 0.6524
          [50]  validation_0-logloss: 0.2841
          [100] validation_0-logloss: 0.1823
          [150] validation_0-logloss: 0.1547
          [187] validation_0-logloss: 0.1498  ← best
          [237] early stopping.

          Arrêté à 187 (sur 500 max) → gain de temps + moins d'overfitting
        OUT
        code_notes: [
          { marker: "learning_rate=0.05", text: "Chaque arbre corrige 5% de l'erreur. Bas = prudent et fin. Règle d'or : baisser le learning_rate ET augmenter n_estimators." },
          { marker: "early_stopping_rounds=50", text: "Si la val_loss ne s'améliore pas pendant 50 arbres → stop. Ici 187 au lieu de 500." },
          { marker: "eval_set", text: "Le validation set pour surveiller. Ce n'est PAS le test set (gardé pour l'évaluation finale)." },
          { marker: "subsample=0.8", text: "Chaque arbre voit 80% des données, tirées aléatoirement. Ajoute de la diversité, réduit l'overfitting." },
        ] },
      { title: "Optimiser les hyperparamètres", explain: "RandomizedSearchCV pour explorer largement, puis GridSearchCV pour affiner.",
        code_block: <<~PY,
          from sklearn.model_selection import RandomizedSearchCV

          params = {
              "max_depth": [3, 5, 7, 9],
              "learning_rate": [0.01, 0.03, 0.05, 0.1],
              "n_estimators": [100, 200, 300, 500],
              "subsample": [0.6, 0.8, 1.0],
              "colsample_bytree": [0.6, 0.8, 1.0],
          }

          search = RandomizedSearchCV(
              XGBClassifier(eval_metric="logloss"), params,
              n_iter=50, cv=5, scoring="f1", n_jobs=-1
          )
          search.fit(X_train, y_train)
          print(f"Best F1 : {search.best_score_:.3f}")
          print(f"Params  : {search.best_params_}")
        PY
        output: <<~OUT,
          Best F1 : 0.938
          Params  : {learning_rate: 0.05, max_depth: 5,
                     n_estimators: 300, subsample: 0.8}
        OUT
        code_notes: [
          { marker: "n_iter=50", text: "50 combinaisons aléatoires. Plus rapide qu'exhaustif." },
        ] },
    ],
    tips: { title: "💡 Best practices XGBoost", items: [
      "Commencer avec lr=0.1, n_estimators=100, puis affiner",
      "XGBoost/LightGBM gèrent les NaN nativement",
      "LE meilleur modèle pour données tabulaires structurées",
      "LightGBM si > 100K lignes (plus rapide)",
    ]},
    code_filename: "xgboost_workflow.py",
    code_content: <<~PY,
      from xgboost import XGBClassifier
      from sklearn.model_selection import train_test_split

      X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2)
      xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                           subsample=0.8, early_stopping_rounds=50, eval_metric="logloss")
      xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)
    PY
  },

  # ============================================================
  # KNN
  # ============================================================
  {
    slug: "knn", title: "K-Nearest Neighbors", icon: "🔵",
    badge: "Supervisé", badge_color: "accent",
    subtitle: "Prédire en regardant les K voisins les plus proches",
    analogy_title: "L'analogie de l'immobilier",
    analogy_text: "Tu veux estimer le prix de ta maison. Tu regardes les 5 maisons les plus similaires vendues récemment et tu fais la moyenne. C'est exactement KNN. K=1 → sensible au bruit. K=50 → trop flou.",
    steps: [
      { title: "Scaling + trouver le meilleur K", explain: "KNN mesure des distances euclidiennes → scaling obligatoire. Le choix de K est le seul vrai hyperparamètre.",
        code_block: <<~PY,
          from sklearn.neighbors import KNeighborsClassifier
          from sklearn.preprocessing import StandardScaler

          scaler = StandardScaler()
          X_train_sc = scaler.fit_transform(X_train)
          X_test_sc  = scaler.transform(X_test)

          train_scores, test_scores = [], []
          for k in range(1, 21):
              knn = KNeighborsClassifier(n_neighbors=k)
              knn.fit(X_train_sc, y_train)
              train_scores.append(knn.score(X_train_sc, y_train))
              test_scores.append(knn.score(X_test_sc, y_test))

          plt.plot(range(1,21), train_scores, "o-", label="Train")
          plt.plot(range(1,21), test_scores, "o-", label="Test")
          plt.xlabel("K"); plt.ylabel("Accuracy"); plt.legend()

          best_k = test_scores.index(max(test_scores)) + 1
          print(f"Meilleur K = {best_k}, accuracy = {max(test_scores):.3f}")
        PY
        output: <<~OUT,
          K=1  Train=1.000  Test=0.842  ← overfitting
          K=3  Train=0.935  Test=0.891
          K=5  Train=0.918  Test=0.905  ← optimal ✓
          K=15 Train=0.878  Test=0.872  ← underfitting
        OUT
        code_notes: [
          { marker: "K=1", text: "Train=100% (chaque point est son propre voisin). Si le voisin est un outlier → catastrophe." },
          { marker: "K=5 optimal", text: "Le coude : score test maximal. En dessous → bruit. Au-dessus → trop de lissage." },
          { marker: "K impair", text: "Pour de la classification binaire, K impair évite les ex-æquo." },
        ] },
    ],
    tips: { title: "⚠️ Limites", items: [
      "Lent en prédiction O(n) — calcule toutes les distances",
      "Malédiction de la dimensionalité : mauvais avec beaucoup de features",
      "Lazy learner : pas de vrai modèle, stocke tout le dataset",
      "Bon pour petits datasets (< 10K) avec peu de features (< 20)",
    ]},
    code_filename: "knn_workflow.py",
    code_content: <<~PY,
      from sklearn.neighbors import KNeighborsClassifier
      from sklearn.preprocessing import StandardScaler

      scaler = StandardScaler()
      X_train_sc = scaler.fit_transform(X_train)
      X_test_sc  = scaler.transform(X_test)
      scores = [(k, KNeighborsClassifier(k).fit(X_train_sc, y_train).score(X_test_sc, y_test))
                for k in range(1, 21)]
      best_k, best_score = max(scores, key=lambda x: x[1])
      print(f"K={best_k}, accuracy={best_score:.3f}")
    PY
  },

  # ============================================================
  # SVM
  # ============================================================
  {
    slug: "svm", title: "Support Vector Machine", icon: "🔲",
    badge: "Supervisé", badge_color: "pink",
    subtitle: "Trouver la frontière optimale entre les classes",
    analogy_title: "L'analogie de la route entre deux villages",
    analogy_text: "SVM trace une route aussi large que possible entre deux villages. C contrôle la tolérance aux erreurs, gamma la portée d'influence. Le kernel RBF projette dans un espace supérieur pour les cas non-linéaires.",
    steps: [
      { title: "Scaling + Grid Search sur C et gamma", explain: "SVM est très sensible à l'échelle. C et gamma interagissent fortement → grid search.",
        code_block: <<~PY,
          from sklearn.svm import SVC
          from sklearn.model_selection import GridSearchCV

          params = {
              "C": [0.1, 1, 10, 100],
              "gamma": ["scale", 0.01, 0.1, 1],
              "kernel": ["rbf", "linear"]
          }

          grid = GridSearchCV(SVC(probability=True), params, cv=5, n_jobs=-1)
          grid.fit(X_train_sc, y_train)

          print(f"Best params : {grid.best_params_}")
          print(f"Best CV     : {grid.best_score_:.3f}")
          print(f"Test        : {grid.score(X_test_sc, y_test):.3f}")
        PY
        output: <<~OUT,
          Best params : {C: 10, gamma: 'scale', kernel: 'rbf'}
          Best CV     : 0.912
          Test        : 0.908
        OUT
        code_notes: [
          { marker: "C=10", text: "C grand → frontière stricte (overfitting). C petit → tolère des erreurs (underfitting)." },
          { marker: "gamma", text: "Gamma grand → influence locale, frontière complexe. Gamma petit → influence large, frontière lisse." },
        ] },
    ],
    tips: { title: "💡 Quand utiliser SVM ?", items: [
      "Bon en haute dimension (n_features > n_samples)",
      "Bon pour petits datasets (< 10K lignes)",
      "Lent au-delà → préférer LinearSVC",
      "⚠️ Scaling OBLIGATOIRE",
    ]},
    code_filename: "svm_workflow.py",
    code_content: <<~PY,
      from sklearn.svm import SVC
      from sklearn.model_selection import GridSearchCV

      params = {"C": [0.1,1,10,100], "gamma": ["scale",0.01,0.1], "kernel": ["rbf","linear"]}
      grid = GridSearchCV(SVC(), params, cv=5, n_jobs=-1)
      grid.fit(X_train_sc, y_train)
      print(f"Best: {grid.best_score_:.3f} — {grid.best_params_}")
    PY
  },

  # ============================================================
  # K-MEANS
  # ============================================================
  {
    slug: "kmeans", title: "K-Means Clustering", icon: "🔮",
    badge: "Non-supervisé", badge_color: "cyan",
    subtitle: "Regrouper automatiquement les données similaires — sans étiquettes",
    analogy_title: "L'analogie des boules de pétanque",
    analogy_text: "Des boules sur un terrain. Tu places K cochonnets au hasard. Chaque boule se rattache au plus proche. Puis tu déplaces chaque cochonnet au centre de ses boules. Répète. Les groupes se stabilisent.",
    steps: [
      { title: "Choisir K — coude + silhouette", explain: "K-Means a besoin de savoir combien de clusters. Le coude montre où ajouter un cluster n'apporte plus grand-chose.",
        code_block: <<~PY,
          from sklearn.cluster import KMeans
          from sklearn.metrics import silhouette_score

          X_sc = StandardScaler().fit_transform(X)

          for k in range(2, 11):
              km = KMeans(n_clusters=k, n_init=10, random_state=42)
              labels = km.fit_predict(X_sc)
              sil = silhouette_score(X_sc, labels)
              print(f"K={k:2d}  inertia={km.inertia_:8.0f}  silhouette={sil:.3f}")
        PY
        output: <<~OUT,
          K= 2  inertia=  12450  silhouette=0.412
          K= 3  inertia=   8230  silhouette=0.485
          K= 4  inertia=   5810  silhouette=0.521 ← max
          K= 5  inertia=   4920  silhouette=0.498
          → K=4 clusters
        OUT
        code_notes: [
          { marker: "inertia_", text: "Somme des distances² au centroïde. Diminue toujours. Le coude est où la pente ralentit." },
          { marker: "silhouette_score", text: "De -1 à 1. Mesure si chaque point est bien dans son cluster. Objectif : maximiser." },
        ] },
      { title: "Analyser et nommer les clusters", explain: "L'algo donne des numéros. C'est à toi de comprendre le profil de chaque groupe.",
        code_block: <<~PY,
          km = KMeans(n_clusters=4, n_init=10, random_state=42)
          df["cluster"] = km.fit_predict(X_sc)
          print(df.groupby("cluster").agg({"age":"mean","income":"mean","purchases":"mean"}).round(1))
        PY
        output: <<~OUT,
               age  income  purchases
          0   28.3  35200      8.2   → "Jeunes actifs"
          1   62.1  42100     15.3   → "Seniors fidèles"
          2   41.5  78400      3.1   → "Gros budgets occasionnels"
          3   45.2  31800      0.8   → "Inactifs à réactiver"
        OUT
        code_notes: [
          { marker: "groupby('cluster')", text: "La moyenne par cluster révèle le profil. C'est la partie data science qui donne la valeur business." },
        ] },
    ],
    tips: { title: "⚠️ Limites", items: [
      "Assume des clusters sphériques",
      "Sensible à l'initialisation → n_init=10",
      "Formes complexes → DBSCAN",
      "⚠️ Scaling obligatoire",
    ]},
    code_filename: "kmeans_workflow.py",
    code_content: <<~PY,
      from sklearn.cluster import KMeans
      from sklearn.preprocessing import StandardScaler
      from sklearn.metrics import silhouette_score

      X_sc = StandardScaler().fit_transform(X)
      for k in range(2, 11):
          km = KMeans(n_clusters=k, n_init=10, random_state=42)
          print(f"K={k} sil={silhouette_score(X_sc, km.fit_predict(X_sc)):.3f}")
    PY
  },

  # ============================================================
  # PCA
  # ============================================================
  {
    slug: "pca", title: "PCA — Composantes Principales", icon: "🎭",
    badge: "Non-supervisé", badge_color: "cyan",
    subtitle: "Réduire les dimensions tout en gardant l'essentiel de l'information",
    analogy_title: "L'analogie de la photo de profil",
    analogy_text: "Une statue 3D photographiée en 2D. La PCA trouve les angles qui captent le maximum de variation. 50 features → peut-être 5 composantes suffisent pour 95% de l'info.",
    steps: [
      { title: "Appliquer et interpréter", explain: "La PCA transforme des features corrélées en composantes indépendantes qui captent la variance maximale.",
        code_block: <<~PY,
          from sklearn.decomposition import PCA
          import numpy as np

          X_sc = StandardScaler().fit_transform(X)

          pca = PCA(n_components=0.95)  # garder 95% de la variance
          X_pca = pca.fit_transform(X_sc)

          print(f"{X_sc.shape[1]} features → {X_pca.shape[1]} composantes")
          for i, var in enumerate(pca.explained_variance_ratio_):
              bar = "█" * int(var * 50)
              print(f"  PC{i+1}: {var:.1%} {bar}")
        PY
        output: <<~OUT,
          18 features → 5 composantes

          PC1: 38.2% ███████████████████
          PC2: 22.1% ███████████
          PC3: 15.4% ████████
          PC4: 11.8% ██████
          PC5:  8.1% ████
          Total: 95.6%
        OUT
        code_notes: [
          { marker: "n_components=0.95", text: "Garde automatiquement assez de composantes pour 95% de la variance. Alternative : <code>PCA(n_components=2)</code> pour visualisation." },
          { marker: "PC1: 38.2%", text: "La 1ère composante capture 38% de la variation. C'est un cocktail des features les plus corrélées." },
        ] },
      { title: "Visualisation 2D", explain: "Projeter des données haute dimension en 2D pour voir les clusters naturels.",
        code_block: <<~PY,
          X_2d = PCA(n_components=2).fit_transform(X_sc)

          plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="viridis", alpha=0.5, s=15)
          plt.xlabel("PC1"); plt.ylabel("PC2")
          plt.colorbar(label="Target")
          plt.title("Projection PCA 2D")
        PY
        output: <<~OUT,
          Si les couleurs forment des zones distinctes → les features
          contiennent de l'information prédictive.
        OUT
        code_notes: [
          { marker: "c=y", text: "Colore par la target. Des zones colorées distinctes = bonne séparabilité." },
        ] },
    ],
    tips: { title: "💡 Cas d'usage", items: [
      "Réduire la dimensionalité avant ML (accélère, réduit overfitting)",
      "Visualisation de données haute dimension en 2D",
      "⚠️ Les composantes ne sont plus interprétables",
      "⚠️ Scaling OBLIGATOIRE",
    ]},
    code_filename: "pca_workflow.py",
    code_content: <<~PY,
      from sklearn.decomposition import PCA
      from sklearn.preprocessing import StandardScaler
      X_sc = StandardScaler().fit_transform(X)
      pca = PCA(n_components=0.95)
      X_pca = pca.fit_transform(X_sc)
      print(f"{X.shape[1]} → {X_pca.shape[1]} composantes")
    PY
  },

  # ============================================================
  # NEURAL NETWORKS
  # ============================================================
  {
    slug: "nn", title: "Neural Networks (Dense)", icon: "🧠",
    badge: "Deep Learning", badge_color: "pink",
    subtitle: "Des couches de neurones qui apprennent des représentations complexes",
    analogy_title: "L'analogie de l'entreprise",
    analogy_text: "Un réseau de neurones = une entreprise à plusieurs étages. Les données entrent à l'accueil. Le 1er étage détecte des patterns simples. Le 2e combine. Le dernier prend la décision. L'entraînement ajuste les poids pour minimiser les erreurs.",
    steps: [
      { title: "Construire et compiler", explain: "On empile des couches Dense avec ReLU. Dropout contre l'overfitting.",
        code_block: <<~PY,
          from tensorflow.keras.models import Sequential
          from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

          model = Sequential([
              Dense(64, activation="relu", input_shape=(X_train_sc.shape[1],)),
              BatchNormalization(),
              Dropout(0.3),           # éteint 30% des neurones
              Dense(32, activation="relu"),
              Dropout(0.2),
              Dense(1, activation="sigmoid")   # sortie binaire
          ])

          model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
          model.summary()
        PY
        output: <<~OUT,
          dense     (None, 64)   1,216 params
          dropout   (None, 64)   0
          dense_1   (None, 32)   2,080
          dense_2   (None, 1)    33
          Total: 3,585 params
        OUT
        code_notes: [
          { marker: "relu", text: "ReLU(x) = max(0, x). Simple et efficace. Résout le vanishing gradient." },
          { marker: "Dropout(0.3)", text: "30% des neurones éteints aléatoirement à chaque batch. Force le réseau à ne pas dépendre de neurones spécifiques." },
          { marker: "sigmoid", text: "Sortie [0,1] → probabilité. Multi-classes : <code>Dense(n, softmax)</code>. Régression : <code>Dense(1)</code> sans activation." },
        ] },
      { title: "Entraîner avec Early Stopping", explain: "Surveiller la val_loss et restaurer les meilleurs poids trouvés.",
        code_block: <<~PY,
          from tensorflow.keras.callbacks import EarlyStopping

          es = EarlyStopping(patience=5, restore_best_weights=True)

          history = model.fit(
              X_train_sc, y_train,
              epochs=100, batch_size=32,
              validation_split=0.2,
              callbacks=[es]
          )

          print(f"Test accuracy : {model.evaluate(X_test_sc, y_test)[1]:.3f}")
          print(f"Stopped at epoch {len(history.history['loss'])}")
        PY
        output: <<~OUT,
          Epoch 1  - val_accuracy: 0.801
          Epoch 10 - val_accuracy: 0.915
          Epoch 20 - val_accuracy: 0.924 ← best
          Epoch 25 - early stopping

          Test accuracy : 0.921
        OUT
        code_notes: [
          { marker: "patience=5", text: "Tolère 5 epochs sans amélioration. 5-10 est un bon compromis." },
          { marker: "restore_best_weights=True", text: "CRUCIAL. Sans ça, le modèle final est celui de la dernière epoch, pas la meilleure." },
        ] },
    ],
    tips: { title: "🏗️ Architecture type", items: [
      "Régression : Dense(1) sans activation, loss=mse",
      "Binaire : Dense(1, sigmoid), loss=binary_crossentropy",
      "Multi-classes : Dense(n, softmax), loss=categorical_crossentropy",
      "Commencer simple (2-3 couches) puis complexifier",
    ]},
    code_filename: "neural_network.py",
    code_content: <<~PY,
      from tensorflow.keras.models import Sequential
      from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
      from tensorflow.keras.callbacks import EarlyStopping

      model = Sequential([
          Dense(64, activation="relu", input_shape=(X_train_sc.shape[1],)),
          BatchNormalization(), Dropout(0.3),
          Dense(32, activation="relu"), Dropout(0.2),
          Dense(1, activation="sigmoid")
      ])
      model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
      model.fit(X_train_sc, y_train, epochs=100, batch_size=32,
                validation_split=0.2, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
    PY
  },

  # ============================================================
  # CNN
  # ============================================================
  {
    slug: "cnn", title: "CNN — Réseaux Convolutifs", icon: "🖼️",
    badge: "Deep Learning", badge_color: "pink",
    subtitle: "Analyser des images par détection de motifs locaux",
    analogy_title: "L'analogie de la loupe",
    analogy_text: "Un CNN regarde une image avec une loupe glissante. La 1re couche détecte bords et couleurs. La suivante combine en contours. Les couches profondes reconnaissent des objets. Le Transfer Learning réutilise un réseau déjà expert.",
    steps: [
      { title: "Préparer les images + Data Augmentation", explain: "Toutes les images à la même taille, normalisées. L'augmentation crée des variantes artificielles.",
        code_block: <<~PY,
          from tensorflow.keras.preprocessing.image import ImageDataGenerator

          train_datagen = ImageDataGenerator(
              rescale=1./255,        # [0,255] → [0,1]
              rotation_range=20,     # rotation ±20°
              horizontal_flip=True,
              zoom_range=0.2,
              validation_split=0.2
          )

          train_gen = train_datagen.flow_from_directory(
              "data/train", target_size=(150, 150),
              batch_size=32, class_mode="binary", subset="training"
          )
        PY
        output: "Found 1600 images belonging to 2 classes",
        code_notes: [
          { marker: "rescale=1./255", text: "Pixels de [0,255] à [0,1]. Les réseaux convergent mieux avec des petites valeurs." },
          { marker: "rotation_range=20", text: "Chaque image tournée aléatoirement → le modèle voit des versions différentes → meilleure généralisation." },
        ] },
      { title: "Transfer Learning (recommandé)", explain: "VGG16 pré-entraîné sur 1.4M d'images + tes couches custom au sommet.",
        code_block: <<~PY,
          from tensorflow.keras.applications import VGG16
          from tensorflow.keras.models import Sequential
          from tensorflow.keras.layers import Flatten, Dense, Dropout

          base = VGG16(weights="imagenet", include_top=False, input_shape=(150,150,3))
          base.trainable = False   # geler les 14.7M params pré-entraînés

          model = Sequential([
              base,
              Flatten(),
              Dense(256, activation="relu"),
              Dropout(0.5),
              Dense(1, activation="sigmoid")
          ])
          model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
          model.fit(train_gen, epochs=30, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
        PY
        output: <<~OUT,
          VGG16 : 14.7M params (gelés)
          Nos couches : 1.2M params (entraînés)

          Epoch 15 - val_accuracy: 0.958 ← best
          Résultat : 95.8% avec ~800 images/classe
          Sans transfer learning : ~75-80%
        OUT
        code_notes: [
          { marker: "base.trainable = False", text: "On gèle VGG16. On n'entraîne que nos couches. Rapide et efficace avec peu de données." },
          { marker: "include_top=False", text: "On enlève les couches de classification ImageNet et on met les nôtres." },
        ] },
    ],
    tips: { title: "🏗️ Architectures pré-entraînées", items: [
      "VGG16 : simple, bon pour le transfer learning pédagogique",
      "ResNet : skip connections, plus profond",
      "MobileNet : léger, idéal pour mobile",
      "EfficientNet : meilleur ratio performance/taille",
    ]},
    code_filename: "cnn_transfer_learning.py",
    code_content: <<~PY,
      from tensorflow.keras.applications import VGG16
      from tensorflow.keras.models import Sequential
      from tensorflow.keras.layers import Flatten, Dense, Dropout

      base = VGG16(weights="imagenet", include_top=False, input_shape=(150,150,3))
      base.trainable = False
      model = Sequential([base, Flatten(), Dense(256,"relu"), Dropout(0.5), Dense(1,"sigmoid")])
      model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    PY
  },

  # ============================================================
  # RNN / NLP
  # ============================================================
  {
    slug: "rnn", title: "RNN / NLP", icon: "📝",
    badge: "Deep Learning", badge_color: "pink",
    subtitle: "Traiter du texte et des séquences",
    analogy_title: "L'analogie du lecteur attentif",
    analogy_text: "Un RNN lit mot par mot avec une mémoire de travail. Le LSTM résout la mémoire courte avec des portes : quoi retenir, quoi oublier.",
    steps: [
      { title: "Preprocessing NLP", explain: "Transformer du texte en séquences de nombres de longueur fixe.",
        code_block: <<~PY,
          from tensorflow.keras.preprocessing.text import Tokenizer
          from tensorflow.keras.preprocessing.sequence import pad_sequences

          tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
          tokenizer.fit_on_texts(X_train_text)

          sequences = tokenizer.texts_to_sequences(X_train_text)
          X_train_pad = pad_sequences(sequences, maxlen=200, padding="post")

          print(f"'Ce film est génial' → {sequences[0]}")
          print(f"Après padding : {X_train_pad[0][:10]}...")
        PY
        output: <<~OUT,
          'Ce film est génial' → [7, 6, 4, 342]
          Après padding : [7, 6, 4, 342, 0, 0, 0, 0, 0, 0]...
        OUT
        code_notes: [
          { marker: "num_words=10000", text: "Garde les 10K mots les plus fréquents. Les rares deviennent <code>&lt;OOV&gt;</code>." },
          { marker: "pad_sequences", text: "Uniformise la longueur. Les phrases courtes sont complétées par des 0." },
        ] },
      { title: "LSTM bidirectionnel", explain: "L'Embedding transforme les numéros en vecteurs denses, le LSTM lit avec mémoire.",
        code_block: <<~PY,
          from tensorflow.keras.models import Sequential
          from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout

          model = Sequential([
              Embedding(10000, 128, input_length=200),
              Bidirectional(LSTM(64)),
              Dropout(0.3),
              Dense(32, activation="relu"),
              Dense(1, activation="sigmoid")
          ])
          model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
          model.fit(X_train_pad, y_train, epochs=20, batch_size=64,
                    validation_split=0.2, callbacks=[EarlyStopping(patience=3)])

          # Prédire
          test = ["Ce film est absolument magnifique"]
          pad = pad_sequences(tokenizer.texts_to_sequences(test), maxlen=200)
          proba = model.predict(pad)[0][0]
          print(f"→ {'positif' if proba > 0.5 else 'négatif'} ({proba:.1%})")
        PY
        output: "→ positif (94.2%)",
        code_notes: [
          { marker: "Embedding", text: "Transforme le nombre 342 ('génial') en vecteur de 128 dimensions capturant le sens." },
          { marker: "Bidirectional", text: "Lit à l'endroit ET à l'envers. Le contexte des deux côtés désambiguïse." },
        ] },
    ],
    tips: { title: "🔑 Choix du modèle NLP", items: [
      "Simple RNN : séquences courtes (vanishing gradient)",
      "LSTM : dépendances longues, le standard",
      "GRU : LSTM simplifié, plus rapide",
      "Transformers (BERT, GPT) : état de l'art, HuggingFace",
    ]},
    code_filename: "rnn_sentiment.py",
    code_content: <<~PY,
      from tensorflow.keras.models import Sequential
      from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
      model = Sequential([Embedding(10000, 128, input_length=200),
                          Bidirectional(LSTM(64)), Dropout(0.3),
                          Dense(32, "relu"), Dense(1, "sigmoid")])
      model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    PY
  },

  # ============================================================
  # PIPELINE SKLEARN
  # ============================================================
  {
    slug: "pipeline", title: "Pipeline Sklearn", icon: "🔧",
    badge: "Architecture", badge_color: "green",
    subtitle: "Enchaîner preprocessing et modèle dans un objet unique",
    analogy_title: "L'analogie de la chaîne de montage",
    analogy_text: "Sans pipeline, tu fais chaque étape à la main. Une Pipeline enchaîne tout automatiquement : imputation → scaling → encoding → modèle. Zéro data leakage, une seule ligne pour fit et predict.",
    steps: [
      { title: "Construire la pipeline", explain: "ColumnTransformer applique le bon traitement par type de colonne, puis on branche le modèle.",
        code_block: <<~PY,
          from sklearn.pipeline import Pipeline
          from sklearn.compose import ColumnTransformer
          from sklearn.impute import SimpleImputer
          from sklearn.preprocessing import StandardScaler, OneHotEncoder
          from sklearn.ensemble import RandomForestClassifier

          num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("scaler", StandardScaler())])
          cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                               ("encoder", OneHotEncoder(handle_unknown="ignore"))])

          preprocessor = ColumnTransformer([
              ("num", num_pipe, ["age", "income", "purchase_count"]),
              ("cat", cat_pipe, ["city", "gender"])
          ])

          pipe = Pipeline([("preprocessing", preprocessor),
                           ("model", RandomForestClassifier(random_state=42))])

          pipe.fit(X_train, y_train)  # UNE SEULE LIGNE
          print(f"Test : {pipe.score(X_test, y_test):.3f}")
        PY
        output: "Test : 0.918",
        code_notes: [
          { marker: "pipe.fit()", text: "Cette seule ligne fait TOUT : imputer, scaler, encoder, entraîner. Le fit_transform vs transform est géré automatiquement." },
          { marker: "ColumnTransformer", text: "Applique le bon traitement selon le type de colonne." },
        ] },
      { title: "GridSearch + sérialisation", explain: "La notation double underscore navigue dans la hiérarchie. joblib sauvegarde tout.",
        code_block: <<~PY,
          from sklearn.model_selection import GridSearchCV
          import joblib

          params = {
              "model__n_estimators": [50, 100, 200],
              "model__max_depth": [5, 10, None],
          }

          grid = GridSearchCV(pipe, params, cv=5, scoring="accuracy", n_jobs=-1)
          grid.fit(X_train, y_train)

          print(f"Best CV  : {grid.best_score_:.3f}")
          print(f"Test     : {grid.score(X_test, y_test):.3f}")

          joblib.dump(grid.best_estimator_, "pipeline.joblib")
          print("✅ Pipeline sauvegardée (preprocessing + modèle)")
        PY
        output: <<~OUT,
          Best CV  : 0.924
          Test     : 0.921
          ✅ Pipeline sauvegardée
        OUT
        code_notes: [
          { marker: "model__n_estimators", text: "Double underscore navigue : <code>model</code> → le RandomForest, <code>n_estimators</code> → son paramètre." },
          { marker: "joblib.dump()", text: "Sauvegarde TOUT : preprocessing + modèle. En production : <code>joblib.load()</code> + <code>pipe.predict()</code>." },
        ] },
    ],
    tips: { title: "🔑 Avantages", items: [
      "Élimine le data leakage automatiquement",
      "Code reproductible (une ligne fit, une ligne predict)",
      "Sérialisable avec joblib",
      "Compatible GridSearchCV",
    ]},
    code_filename: "sklearn_pipeline.py",
    code_content: <<~PY,
      from sklearn.pipeline import Pipeline
      from sklearn.compose import ColumnTransformer
      from sklearn.impute import SimpleImputer
      from sklearn.preprocessing import StandardScaler, OneHotEncoder
      from sklearn.ensemble import RandomForestClassifier
      import joblib

      num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
      cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("encoder", OneHotEncoder(handle_unknown="ignore"))])
      preprocessor = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
      pipe = Pipeline([("preprocessing", preprocessor), ("model", RandomForestClassifier())])
      pipe.fit(X_train, y_train)
      joblib.dump(pipe, "pipeline.joblib")
    PY
  },

  # ============================================================
  # MLOPS
  # ============================================================
  {
    slug: "mlops", title: "MLOps — Déploiement", icon: "☁️",
    badge: "Production", badge_color: "orange",
    subtitle: "Du notebook au modèle en production accessible par API",
    analogy_title: "L'analogie du restaurant",
    analogy_text: "Ton notebook = la recette chez toi. Le déploiement = ouvrir un restaurant. Il faut : emballer la recette (joblib), créer le service (FastAPI), construire la cuisine (Docker), installer le lieu (Cloud Run).",
    steps: [
      { title: "API avec FastAPI", explain: "L'interface entre le monde et ton modèle. Envoi HTTP → prédiction JSON.",
        code_block: <<~PY,
          # api.py
          from fastapi import FastAPI
          import joblib
          import pandas as pd

          app = FastAPI()
          pipe = joblib.load("pipeline.joblib")

          @app.get("/predict")
          def predict(age: int, income: float, city: str):
              X = pd.DataFrame([{"age": age, "income": income,
                                 "purchase_count": 0, "city": city, "gender": "M"}])
              pred = pipe.predict(X)[0]
              proba = pipe.predict_proba(X)[0].max()
              return {"prediction": int(pred), "confidence": round(float(proba), 3)}

          # Lancer : uvicorn api:app --reload
        PY
        output: <<~OUT,
          $ curl "localhost:8000/predict?age=35&income=55000&city=Nancy"

          {"prediction": 1, "confidence": 0.873}
        OUT
        code_notes: [
          { marker: "@app.get('/predict')", text: "Endpoint GET. FastAPI type les paramètres automatiquement." },
          { marker: "pipe.predict(X)", text: "La pipeline fait TOUT : imputer, scaler, encoder, prédire." },
        ] },
      { title: "Dockeriser et déployer", explain: "Docker = boîte hermétique. Cloud Run = serverless auto-scaling.",
        code_block: <<~PY,
          # Dockerfile
          # FROM python:3.10-slim
          # WORKDIR /app
          # COPY requirements.txt .
          # RUN pip install -r requirements.txt
          # COPY . .
          # CMD uvicorn api:app --host 0.0.0.0 --port $PORT

          # Build & test
          # docker build -t my-model .
          # docker run -p 8000:8000 -e PORT=8000 my-model

          # Deploy GCP Cloud Run
          # gcloud builds submit --tag gcr.io/PROJECT/my-model
          # gcloud run deploy --image gcr.io/PROJECT/my-model --allow-unauthenticated
        PY
        output: <<~OUT,
          ✅ https://my-model-xxxxx.run.app
          0 requêtes = 0 €
          1000 req/sec = auto-scaling
          HTTPS automatique
        OUT
        code_notes: [
          { marker: "python:3.10-slim", text: "Image légère ~150 MB. L'image complète ~900 MB." },
          { marker: "--port $PORT", text: "Cloud Run injecte le port. Ne pas hardcoder." },
        ] },
    ],
    tips: { title: "📋 Checklist", items: [
      "requirements.txt avec versions FIGÉES (==)",
      ".env pour les secrets (jamais dans le code !)",
      "Tester en local avant de déployer",
      "Modèle sur GCS plutôt que dans l'image Docker",
    ]},
    code_filename: "mlops_deploy.py",
    code_content: <<~PY,
      from fastapi import FastAPI
      import joblib, pandas as pd

      app = FastAPI()
      pipe = joblib.load("pipeline.joblib")

      @app.get("/predict")
      def predict(age: int, income: float, city: str):
          X = pd.DataFrame([{"age": age, "income": income, "city": city}])
          return {"prediction": int(pipe.predict(X)[0]),
                  "confidence": float(pipe.predict_proba(X)[0].max())}
    PY
  },
)
