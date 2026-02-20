# frozen_string_literal: true
# All 15 Le Wagon DS workflows with detailed code, outputs, and annotations
# Using Ruby heredocs = zero indentation issues

WORKFLOWS = [
  {
    slug: "preprocessing",
    title: "Data Preprocessing",
    icon: "🧹",
    badge: "Préparation",
    badge_color: "green",
    subtitle: "Nettoyer et transformer les données brutes en un dataset exploitable par un modèle",
    analogy_title: "L'analogie du chef cuisinier",
    analogy_text: "Le preprocessing, c'est comme la mise en place en cuisine. Avant de cuisiner (entraîner un modèle), il faut laver les légumes (supprimer les doublons), retirer les parties abîmées (gérer les valeurs manquantes), tout couper à la même taille (scaling) et séparer les ingrédients par type (encodage). Si tu sautes cette étape, le plat sera raté, même avec la meilleure recette.",
    steps: [
      {
        title: "Charger et inspecter les données",
        explain: "On charge le CSV et on prend la température du dataset. Combien de lignes ? De colonnes ? Quels types ? Des valeurs aberrantes visibles dès le describe() ?",
        code_block: <<~PY,
          import pandas as pd

          df = pd.read_csv("housing.csv")
          print(df.shape)
          print(df.dtypes)
          print(df.describe())
          print(df.head(3))
        PY
        output: <<~OUT,
          (1460, 7)

          surface     int64
          rooms       int64
          age         int64
          city       object
          garden     object
          pool       object
          price       int64

               surface  rooms    age     price
          mean   102.3   3.8   22.1   245_800
          min     18.0   1.0    0.0    45_000
          max    450.0  12.0   95.0  1_200_000

             surface  rooms  age    city     garden  pool    price
          0       85      3   15    Paris     Oui    Non   320000
          1      120      5   30    Lyon      Non    Non   185000
          2       45      2    5    Marseille Oui    Oui   142000
        OUT
        code_notes: [
          { marker: "df.shape", text: "Donne (lignes, colonnes). Ici 1460 observations et 7 variables. C'est le premier réflexe." },
          { marker: "df.dtypes", text: "Vérifie les types. <code>city</code> est <code>object</code> (texte) → il faudra l'encoder. <code>price</code> est <code>int64</code> → c'est notre target numérique." },
          { marker: "describe()", text: "Résumé statistique. Un <code>age</code> min à 0 est suspect (maison neuve ou erreur ?). Un <code>surface</code> max à 450 m² est-il un outlier ?" },
        ]
      },
      {
        title: "Gérer les valeurs manquantes",
        explain: "Des cases vides dans le dataset. On les détecte, on comprend le pattern (aléatoire ou systématique ?), puis on les traite : suppression ou remplissage intelligent.",
        code_block: <<~PY,
          # Compter les NaN par colonne
          print(df.isnull().sum())
          print(f"\\n% manquants :\\n{(df.isnull().mean() * 100).round(1)}")

          # Stratégie selon le % de manquants
          # < 5%  → remplir (imputer)
          # > 60% → supprimer la colonne
          # Entre → analyser le pattern

          from sklearn.impute import SimpleImputer

          # Numériques → médiane (robuste aux outliers)
          imputer_num = SimpleImputer(strategy="median")
          df["surface"] = imputer_num.fit_transform(df[["surface"]])

          # Catégorielles → valeur la plus fréquente
          imputer_cat = SimpleImputer(strategy="most_frequent")
          df["city"] = imputer_cat.fit_transform(df[["city"]]).ravel()
        PY
        output: <<~OUT,
          surface     23
          rooms        0
          age         45
          city        12
          garden       8
          pool         0
          price        0

          % manquants :
          surface    1.6
          rooms      0.0
          age        3.1
          city       0.8
          garden     0.5
          pool       0.0
          price      0.0
        OUT
        code_notes: [
          { marker: "isnull().sum()", text: "Compte les NaN par colonne. Ici <code>age</code> a 45 valeurs manquantes (3.1 %). C'est peu → on impute." },
          { marker: "strategy='median'", text: "Pourquoi la médiane et pas la moyenne ? Parce que la médiane est robuste aux outliers. Si tu as des surfaces de 18 à 450 m², la moyenne sera tirée vers le haut." },
          { marker: "most_frequent", text: "Pour les catégories, on remplit avec la valeur la plus courante. Si 60 % des lignes ont <code>city='Paris'</code>, les NaN deviennent <code>Paris</code>." },
        ]
      },
      {
        title: "Supprimer les doublons",
        explain: "Des lignes identiques qui faussent la distribution. Le modèle pensera que ce profil est plus fréquent qu'il ne l'est.",
        code_block: <<~PY,
          print(f"Doublons : {df.duplicated().sum()}")

          # Voir les doublons
          print(df[df.duplicated(keep=False)].sort_values("price").head(4))

          # Supprimer
          df = df.drop_duplicates()
          print(f"Shape après : {df.shape}")
        PY
        output: <<~OUT,
          Doublons : 17

             surface  rooms  age  city   garden pool   price
          42      85      3   15  Paris   Oui   Non  320000
          98      85      3   15  Paris   Oui   Non  320000
          67     120      5   30  Lyon    Non   Non  185000
          201    120      5   30  Lyon    Non   Non  185000

          Shape après : (1443, 7)
        OUT
        code_notes: [
          { marker: "duplicated()", text: "Renvoie <code>True</code> pour chaque ligne qui est une copie exacte d'une autre. <code>keep=False</code> marque TOUTES les copies (pas juste la 2e)." },
          { marker: "drop_duplicates()", text: "Supprime les doublons, garde la première occurrence. 17 lignes en moins ici." },
        ]
      },
      {
        title: "Détecter et traiter les outliers",
        explain: "Un élève qui mesure '17 mètres' est clairement une erreur de saisie. Les outliers faussent la moyenne et le modèle — on les repère et on décide quoi en faire.",
        code_block: <<~PY,
          import numpy as np

          # Méthode IQR (Inter-Quartile Range)
          Q1 = df["surface"].quantile(0.25)   # 25e percentile
          Q3 = df["surface"].quantile(0.75)   # 75e percentile
          IQR = Q3 - Q1

          lower = Q1 - 1.5 * IQR
          upper = Q3 + 1.5 * IQR

          outliers = df[(df["surface"] < lower) | (df["surface"] > upper)]
          print(f"Bornes : [{lower:.0f}, {upper:.0f}]")
          print(f"Outliers surface : {len(outliers)}")

          # Option 1 : Supprimer
          df_clean = df[(df["surface"] >= lower) & (df["surface"] <= upper)]

          # Option 2 : Capper (remplacer par la borne)
          df["surface"] = df["surface"].clip(lower, upper)
        PY
        output: <<~OUT,
          Bornes : [12, 238]
          Outliers surface : 23

          Exemples d'outliers :
             surface  rooms  age   city      price
          88     450     12   10   Paris   1200000   ← château ?
          342      5      1   80   Marseille  45000  ← studio 5 m² ?
        OUT
        code_notes: [
          { marker: "IQR", text: "L'écart entre le 1er et 3e quartile. 50 % des données sont dans cet intervalle. On considère comme outlier tout ce qui dépasse 1.5× l'IQR au-delà." },
          { marker: "clip()", text: "Au lieu de supprimer, on plafonne/plancher. Un 450 m² devient 238 m². Moins de perte de données, mais attention à ne pas déformer la distribution." },
        ]
      },
      {
        title: "Encoder les variables catégorielles",
        explain: "Un modèle ML ne comprend que les chiffres. Il faut transformer 'Paris', 'Lyon', 'Marseille' en nombres — mais intelligemment.",
        code_block: <<~PY,
          from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

          # CAS 1 — Ordinales (il y a un ordre naturel)
          # Ex: taille = "S" < "M" < "L" < "XL"
          oe = OrdinalEncoder(categories=[["S", "M", "L", "XL"]])
          df["taille_encoded"] = oe.fit_transform(df[["taille"]])
          # S → 0, M → 1, L → 2, XL → 3

          # CAS 2 — Nominales (pas d'ordre)
          # Ex: city = Paris, Lyon, Marseille
          ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
          encoded = ohe.fit_transform(df[["city"]])
          print(ohe.get_feature_names_out())
          print(encoded[:3])
        PY
        output: <<~OUT,
          ['city_Lyon' 'city_Marseille' 'city_Nancy' 'city_Paris']

          [[0. 0. 0. 1.]     ← Paris
           [1. 0. 0. 0.]     ← Lyon
           [0. 1. 0. 0.]]    ← Marseille
        OUT
        code_notes: [
          { marker: "OrdinalEncoder", text: "Pour les variables avec un ORDRE. Taille S < M < L → 0, 1, 2. Le modèle comprend que L > S." },
          { marker: "OneHotEncoder", text: "Pour les variables SANS ordre. Créer <code>city=1, Lyon=2</code> impliquerait que Lyon > Paris. Le One-Hot crée une colonne binaire par catégorie." },
          { marker: "handle_unknown='ignore'", text: "Si une ville inconnue apparaît en production, au lieu de crasher, le modèle mettra 0 partout. Indispensable pour la robustesse." },
        ]
      },
      {
        title: "Feature Engineering",
        explain: "L'art de créer des variables plus informatives à partir des existantes. Un bon feature engineering vaut souvent plus qu'un modèle sophistiqué.",
        code_block: <<~PY,
          # Prix au m² (plus informatif que le prix brut)
          df["price_per_m2"] = df["price"] / df["surface"]

          # Âge de la maison (à partir de l'année de construction)
          df["age"] = 2026 - df["year_built"]

          # Variables booléennes combinées
          df["has_outdoor"] = ((df["garden"] == "Oui") | (df["pool"] == "Oui")).astype(int)

          # Extraction depuis une date
          df["sale_date"] = pd.to_datetime(df["sale_date"])
          df["sale_month"] = df["sale_date"].dt.month
          df["is_summer"]  = df["sale_month"].isin([6,7,8]).astype(int)

          print(df[["surface", "price", "price_per_m2", "has_outdoor"]].head(3))
        PY
        output: <<~OUT,
             surface   price  price_per_m2  has_outdoor
          0       85  320000       3765           1
          1      120  185000       1542           0
          2       45  142000       3156           1
        OUT
        code_notes: [
          { marker: "price_per_m2", text: "Un appart de 120 m² à 185K (1542 €/m²) est très différent d'un 45 m² à 142K (3156 €/m²). Le ratio donne une info que le modèle ne peut pas calculer seul." },
          { marker: "is_summer", text: "Les prix immobiliers sont souvent plus élevés en été. Transformer un mois en saison binaire aide le modèle à capturer cette saisonnalité." },
        ]
      },
      {
        title: "Scaling / Normalisation",
        explain: "Remettre toutes les variables numériques à la même échelle. Indispensable pour KNN, SVM, réseaux de neurones, régression logistique. Les arbres s'en fichent.",
        code_block: <<~PY,
          from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

          # StandardScaler : moyenne=0, écart-type=1
          scaler = StandardScaler()
          X_train_sc = scaler.fit_transform(X_train[["surface", "age", "rooms"]])

          # ⚠️ RÈGLE D'OR : fit sur TRAIN, transform sur TEST
          X_test_sc = scaler.transform(X_test[["surface", "age", "rooms"]])

          print("Avant scaling :")
          print(f"  surface : mean={X_train['surface'].mean():.0f}, std={X_train['surface'].std():.0f}")
          print("Après scaling :")
          print(f"  surface : mean={X_train_sc[:,0].mean():.2f}, std={X_train_sc[:,0].std():.2f}")
        PY
        output: <<~OUT,
          Avant scaling :
            surface : mean=102, std=48
            age     : mean=22,  std=15
            rooms   : mean=4,   std=2

          Après scaling :
            surface : mean=0.00, std=1.00
            age     : mean=0.00, std=1.00
            rooms   : mean=0.00, std=1.00
        OUT
        code_notes: [
          { marker: "fit_transform(X_train)", text: "<code>fit</code> calcule la moyenne et l'écart-type du train. <code>transform</code> applique (x - mean) / std. Les deux en une ligne." },
          { marker: "transform(X_test)", text: "⚠️ On ne fait PAS <code>fit_transform</code> sur le test ! On réutilise la moyenne et std du train. Sinon, on 'triche' en regardant les données de test." },
          { marker: "StandardScaler", text: "Choix : <code>StandardScaler</code> (normal), <code>MinMaxScaler</code> (borné [0,1], images), <code>RobustScaler</code> (résistant aux outliers, utilise médiane/IQR)." },
        ]
      },
      {
        title: "Train / Test Split",
        explain: "Séparer en deux : un jeu pour apprendre, un pour évaluer. Comme séparer les questions d'un examen en révision et examen blanc.",
        code_block: <<~PY,
          from sklearn.model_selection import train_test_split

          X = df.drop(columns=["price"])
          y = df["price"]

          X_train, X_test, y_train, y_test = train_test_split(
              X, y,
              test_size=0.3,     # 30% pour le test
              random_state=42    # reproductibilité
          )

          print(f"Train : {X_train.shape[0]} lignes ({X_train.shape[0]/len(X)*100:.0f}%)")
          print(f"Test  : {X_test.shape[0]} lignes ({X_test.shape[0]/len(X)*100:.0f}%)")
        PY
        output: <<~OUT,
          Train : 1010 lignes (70%)
          Test  : 433 lignes (30%)
        OUT
        code_notes: [
          { marker: "test_size=0.3", text: "70/30 est standard. Pour un petit dataset (< 1000), on peut faire 80/20." },
          { marker: "random_state=42", text: "Fixe le hasard pour la reproductibilité. Sans ça, chaque exécution donne un split différent." },
          { marker: "⚠️ Ordre", text: "TOUJOURS splitter AVANT le scaling. Si tu scales avant, le scaler a vu les données de test → data leakage." },
        ]
      },
    ],
    tips: { title: "⚠️ Pièges courants", items: [
      "Ne JAMAIS fit le scaler sur le test set → fit_transform() sur train, transform() sur test",
      "OneHotEncoder peut exploser la dimensionalité (100 villes = 100 colonnes)",
      "Les arbres de décision n'ont PAS besoin de scaling",
      "Vérifier les types : un 'code postal' numérique est en réalité catégoriel",
    ]},
    code_filename: "preprocessing_pipeline_complet.py",
    code_content: <<~PY,
      import pandas as pd
      from sklearn.model_selection import train_test_split
      from sklearn.impute import SimpleImputer
      from sklearn.preprocessing import StandardScaler, OneHotEncoder
      from sklearn.compose import ColumnTransformer
      from sklearn.pipeline import Pipeline

      df = pd.read_csv("housing.csv")
      df = df.drop_duplicates()
      X = df.drop(columns=["price"])
      y = df["price"]

      num_cols = X.select_dtypes(include="number").columns.tolist()
      cat_cols = X.select_dtypes(include="object").columns.tolist()

      num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
      cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("encoder", OneHotEncoder(handle_unknown="ignore"))])
      preprocessor = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
      X_train_processed = preprocessor.fit_transform(X_train)
      X_test_processed  = preprocessor.transform(X_test)
      print(f"Train: {X_train_processed.shape}, Test: {X_test_processed.shape}")
    PY
  },

  # ============================================================
  # EDA
  # ============================================================
  {
    slug: "eda",
    title: "Exploratory Data Analysis",
    icon: "📊",
    badge: "Analyse",
    badge_color: "cyan",
    subtitle: "Comprendre la structure, les distributions et les relations dans les données",
    analogy_title: "L'analogie du détective",
    analogy_text: "L'EDA, c'est l'enquête préliminaire avant de résoudre l'affaire. Tu ne lances pas de modèle à l'aveugle : tu regardes les indices (distributions), tu cherches des corrélations (qui connaît qui ?), tu repères les anomalies. Plus ton enquête est minutieuse, plus ton modèle sera pertinent.",
    steps: [
      {
        title: "Vue d'ensemble",
        explain: "Le premier coup d'œil. Combien de lignes et colonnes ? Quels types ? Des valeurs manquantes ?",
        code_block: <<~PY,
          import pandas as pd
          df = pd.read_csv("housing.csv")

          print(f"Shape : {df.shape}")
          print(f"\\nTypes :\\n{df.dtypes}")
          print(f"\\nValeurs manquantes :\\n{df.isnull().sum()}")
          print(f"\\nStats :\\n{df.describe().round(1)}")
        PY
        output: <<~OUT,
          Shape : (1460, 7)

          Types :
          surface     int64
          rooms       int64
          age         int64
          city       object
          price       int64

          Stats :
                  surface  rooms    age      price
          count  1437.0   1460.0  1415.0   1460.0
          mean    102.3      3.8    22.1  245800.0
          min      18.0      1.0     0.0   45000.0
          50%      92.0      3.0    20.0  215000.0
          max     450.0     12.0    95.0 1200000.0
        OUT
        code_notes: [
          { marker: "describe()", text: "Le min/max révèle les outliers. Ici <code>surface=450</code> et <code>age=95</code> sont suspects. Le <code>50%</code> (médiane) donne le centre réel." },
        ]
      },
      {
        title: "Distributions (analyse univariée)",
        explain: "Chaque variable individuellement. Les prix suivent-ils une gaussienne ? Y a-t-il des pics ? Ça conditionne le scaling et les transformations.",
        code_block: <<~PY,
          import matplotlib.pyplot as plt
          import seaborn as sns
          import numpy as np

          fig, axes = plt.subplots(2, 2, figsize=(12, 8))

          # Histogramme + KDE
          sns.histplot(df["price"], bins=30, kde=True, ax=axes[0,0])
          axes[0,0].set_title("Distribution des prix")
          axes[0,0].axvline(df["price"].median(), color="red", linestyle="--")

          # Boxplot (outliers visibles)
          sns.boxplot(x=df["surface"], ax=axes[0,1])
          axes[0,1].set_title("Boxplot surface")

          # Countplot catégorielle
          sns.countplot(y=df["city"], order=df["city"].value_counts().index, ax=axes[1,0])

          # Log-transform si skewed
          sns.histplot(df["price"].apply(np.log1p), bins=30, ax=axes[1,1])
          axes[1,1].set_title("log(price) — plus gaussien")
          plt.tight_layout()
        PY
        output: <<~OUT,
          📊 Distribution des prix :
          ██████████████████████████████  ← pic autour de 200K
          ████████████████████████
          ██████████████████
          ████████████                    ← longue traîne à droite
          ██████
          ███
          █                               ← quelques villas > 800K

          → Distribution skewed → envisager log(price) comme target
        OUT
        code_notes: [
          { marker: "kde=True", text: "Ajoute une courbe lissée sur l'histogramme. Permet de voir la forme de la distribution sans dépendre du nombre de bins." },
          { marker: "log1p()", text: "Si la distribution est très asymétrique, le log la rend plus gaussienne. Beaucoup de modèles fonctionnent mieux avec des distributions symétriques." },
        ]
      },
      {
        title: "Relations entre variables (bivariée)",
        explain: "Surface et prix bougent-ils ensemble ? Les maisons avec jardin sont-elles plus chères ? C'est ici qu'on identifie les features à fort pouvoir prédictif.",
        code_block: <<~PY,
          fig, axes = plt.subplots(1, 3, figsize=(15, 4))

          # Scatter : surface vs price
          axes[0].scatter(df["surface"], df["price"], alpha=0.3, s=10)
          axes[0].set_xlabel("Surface (m²)")
          axes[0].set_ylabel("Prix (€)")
          axes[0].set_title("Surface vs Prix")

          # Boxplot groupé : prix par ville
          sns.boxplot(data=df, x="city", y="price", ax=axes[1])
          axes[1].set_title("Prix par ville")

          # Pairplot
          sns.pairplot(df[["surface", "rooms", "age", "price"]], corner=True)
        PY
        output: <<~OUT,
          📊 Surface vs Prix :
          Relation quasi-linéaire visible.
          surface sera la feature la plus prédictive.
          Paris nettement plus cher que Lyon et Marseille.
        OUT
        code_notes: [
          { marker: "alpha=0.3", text: "Semi-transparence. Avec 1460 points, sans ça on voit une bouillie. Avec, on distingue les zones denses." },
          { marker: "pairplot(corner=True)", text: "Tous les scatter plots 2 à 2 en une commande. <code>corner=True</code> évite la redondance." },
        ]
      },
      {
        title: "Matrice de corrélation",
        explain: "La carte thermique qui résume toutes les relations numériques en un coup d'œil.",
        code_block: <<~PY,
          corr = df.corr(numeric_only=True)

          plt.figure(figsize=(8, 6))
          sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
                      vmin=-1, vmax=1, center=0)
          plt.title("Matrice de corrélation")

          # Top corrélations avec la target
          print(corr["price"].sort_values(ascending=False))
        PY
        output: <<~OUT,
          Corrélations avec price :
          surface    0.81  ← forte !
          rooms      0.68
          age       -0.35  ← négatif (vieux = moins cher)

          ⚠️ surface & rooms corrélés à 0.72 → multicolinéarité
          → Surveiller en régression linéaire (VIF)
        OUT
        code_notes: [
          { marker: "annot=True", text: "Affiche les coefficients directement sur la heatmap." },
          { marker: "corr > 0.8", text: "Deux features corrélées à > 0.8 → multicolinéarité. En régression linéaire, ça rend les coefficients instables." },
          { marker: "corr['price']", text: "Les corrélations avec la target. <code>surface: 0.81</code> est la feature la plus prédictive." },
        ]
      },
      {
        title: "Analyse des valeurs manquantes",
        explain: "Les données manquent-elles au hasard ou y a-t-il un pattern ? C'est crucial pour choisir la bonne stratégie d'imputation.",
        code_block: <<~PY,
          import missingno as msno  # pip install missingno

          msno.matrix(df)
          plt.savefig("missing_patterns.png")

          # Types de manquance :
          # MCAR — Missing Completely At Random (pas de pattern)
          #        → safe to drop or impute mean/median
          # MAR  — Missing At Random (dépend d'une AUTRE variable)
          #        → imputer en fonction de cette variable
          # MNAR — Missing Not At Random (dépend de la variable elle-même)
          #        → le plus dangereux
        PY
        output: <<~OUT,
          surface ████████████████████░███████  (23 NaN)
          rooms   █████████████████████████████  (0 NaN)
          age     █████████████████░░██████████  (45 NaN)
          city    ████████████████████████████░  (12 NaN)
          price   █████████████████████████████  (0 NaN)

          → NaN dispersés (MCAR probable) → safe pour imputer médiane
        OUT
        code_notes: [
          { marker: "msno.matrix()", text: "Visualise les patterns. Si les trous s'alignent entre colonnes, c'est du MAR (dépendance entre variables)." },
          { marker: "MNAR", text: "Le cas vicieux : les données manquent PARCE QUE leur valeur est extrême. Ex: les maisons très chères n'affichent pas leur prix." },
        ]
      },
      {
        title: "Synthèse et stratégie",
        explain: "Le rapport d'enquête final : insights clés et plan de modélisation.",
        code_block: <<~PY,
          # SYNTHÈSE EDA — housing.csv
          # =============================
          # 1. Target (price) : skewed → tester log(price)
          # 2. Feature #1 : surface (corr=0.81) — relation linéaire
          # 3. Feature #2 : rooms (corr=0.68) — corrélé à surface
          # 4. Feature #3 : age (corr=-0.35) — faible mais utile
          # 5. Catégorielle : city — OneHot (4 villes)
          # 6. Outliers : 23 observations surface > 238 m²
          # 7. NaN : < 5% partout → imputer median/most_frequent
          #
          # PLAN :
          # - Baseline : LinearRegression
          # - Tester : Ridge/Lasso (multicolinéarité)
          # - Tester : RandomForest (non-linéarités)
          # - Métrique : RMSE (en €)
        PY
        output: "",
        code_notes: [
          { marker: "synthèse", text: "Un bon EDA se termine par un plan d'action clair. L'EDA n'est pas une fin en soi — c'est le brief pour la modélisation." },
        ]
      },
    ],
    tips: { title: "📦 Toolbox visualisation", items: [
      "matplotlib : contrôle fin, base de tout",
      "seaborn : statistiques visuelles élégantes (histplot, heatmap, pairplot)",
      "plotly : interactif, idéal pour présentation",
      "ydata-profiling : EDA automatisé en 1 ligne — ProfileReport(df)",
    ]},
    code_filename: "eda_complet.py",
    code_content: <<~PY,
      import pandas as pd
      import seaborn as sns
      import matplotlib.pyplot as plt
      import numpy as np

      df = pd.read_csv("housing.csv")
      print(df.shape, df.dtypes, df.describe(), sep="\\n\\n")
      df.hist(figsize=(14, 10), bins=30); plt.tight_layout(); plt.show()
      sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f"); plt.show()
      print(df.corr(numeric_only=True)["price"].sort_values(ascending=False))
    PY
  },
]
