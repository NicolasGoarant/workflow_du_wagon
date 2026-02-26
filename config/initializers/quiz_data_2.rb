# frozen_string_literal: true
# Additional quiz questions (104-200)

QUIZ_QUESTIONS.push(

  # ── NON-SUPERVISÉ : K-MEANS (12 questions) ──

  { id: 104, category: "Non-supervisé", difficulty: "facile",
    question: "Quelle est la différence fondamentale entre supervisé et non-supervisé ?",
    choices: ["Le supervisé est plus rapide", "En non-supervisé, il n'y a PAS de target (y) — on cherche des patterns dans les données", "Le non-supervisé utilise plus de données", "Le supervisé ne marche qu'en classification"],
    answer: 1,
    explanation: "En supervisé on a des étiquettes (y) : spam/pas spam, prix en €. En non-supervisé, on a juste X et on cherche des structures cachées : groupes, axes principaux, anomalies." },

  { id: 105, category: "Non-supervisé", difficulty: "facile",
    question: "Que fait K-Means ?",
    choices: ["Prédit une valeur continue", "Regroupe les données en K clusters en minimisant la distance intra-cluster", "Réduit les dimensions", "Classifie avec des labels connus"],
    answer: 1,
    explanation: "K-Means place K centroïdes, assigne chaque point au centroïde le plus proche, puis recalcule les centroïdes. On répète jusqu'à convergence. Résultat : K groupes homogènes." },

  { id: 106, category: "Non-supervisé", difficulty: "moyen",
    question: "Comment choisir le nombre K de clusters ?",
    choices: ["Toujours K=3", "La méthode du coude (Elbow) : tracer l'inertie en fonction de K et chercher le 'coude'", "K = nombre de features", "K = nombre de lignes / 100"],
    answer: 1,
    explanation: "On teste K=2, 3, 4... et on trace l'inertie. L'inertie baisse toujours, mais il y a un 'coude' où le gain ralentit. Ce coude indique le bon K. On peut compléter avec le silhouette score." },

  { id: 107, category: "Non-supervisé", difficulty: "moyen",
    question: "Qu'est-ce que l'<strong>inertie</strong> en K-Means ?",
    choices: ["La vitesse de convergence", "La somme des distances au carré entre chaque point et son centroïde", "Le nombre d'itérations", "Le nombre de clusters"],
    answer: 1,
    explanation: "Inertie = Σ ||xᵢ − centroïde||². Plus l'inertie est basse, plus les clusters sont compacts. Mais inertie = 0 arrive avec K = nombre de points (chaque point est son propre cluster) → pas utile." },

  { id: 108, category: "Non-supervisé", difficulty: "moyen",
    question: "K-Means est sensible à l'initialisation des centroïdes. Comment sklearn résout-il ce problème ?",
    choices: ["Il n'y a pas de solution", "L'option init='k-means++' place les centroïdes intelligemment, puis n_init=10 fait 10 essais", "Il utilise toujours les mêmes centroïdes", "Il supprime les outliers d'abord"],
    answer: 1,
    explanation: "k-means++ espace les centroïdes initiaux pour éviter qu'ils soient proches. Puis n_init=10 lance K-Means 10 fois avec des initialisations différentes et garde le meilleur résultat (inertie la plus basse)." },

  { id: 109, category: "Non-supervisé", difficulty: "difficile",
    question: "Le <strong>silhouette score</strong> d'un point est de -0.3. Que signifie-t-il ?",
    choices: ["Le point est bien placé", "Le point est probablement dans le MAUVAIS cluster — plus proche d'un autre cluster que du sien", "C'est un outlier à supprimer", "Le score est invalide"],
    answer: 1,
    explanation: "Silhouette = (b − a) / max(a, b) où a = distance moyenne au sein du cluster, b = distance moyenne au cluster le plus proche. Score négatif → a > b → le point est plus proche d'un autre cluster. Score entre -1 (mauvais) et +1 (parfait)." },

  { id: 110, category: "Non-supervisé", difficulty: "facile",
    question: "Pourquoi faut-il scaler les données AVANT K-Means ?",
    choices: ["Pour accélérer le calcul", "K-Means utilise la distance euclidienne — une feature en km dominerait une feature en m", "Parce que sklearn l'exige", "C'est optionnel"],
    answer: 1,
    explanation: "Si 'revenu' va de 20K à 200K et 'âge' de 18 à 80, les distances seront dominées par le revenu. Le scaling ramène tout à la même échelle pour que chaque feature pèse de façon équitable." },

  { id: 111, category: "Non-supervisé", difficulty: "moyen",
    question: "K-Means produit toujours des clusters de forme sphérique. Quand est-ce un problème ?",
    choices: ["Jamais", "Quand les vrais groupes ont des formes allongées, en anneaux ou de tailles très différentes", "Seulement en 3D", "Quand K est pair"],
    answer: 1,
    explanation: "K-Means utilise la distance au centroïde → clusters convexes et sphériques. Pour des formes irrégulières, utiliser DBSCAN (basé sur la densité) ou Gaussian Mixture Models." },

  { id: 112, category: "Non-supervisé", difficulty: "difficile",
    question: "Qu'est-ce que <strong>DBSCAN</strong> et quel avantage a-t-il sur K-Means ?",
    choices: ["Un type de réseau de neurones", "Un clustering basé sur la densité qui détecte les clusters de forme arbitraire et les outliers, sans choisir K", "Un scaler plus rapide", "Un random forest non-supervisé"],
    answer: 1,
    explanation: "DBSCAN regroupe les points denses et marque les points isolés comme bruit (-1). Avantages : pas besoin de choisir K, détecte les outliers, clusters de toute forme. Inconvénient : deux hyperparamètres (eps, min_samples) à régler." },

  { id: 113, category: "Non-supervisé", difficulty: "moyen",
    question: "Comment évaluer un clustering quand on N'A PAS de labels ?",
    choices: ["Impossible sans labels", "Avec des métriques internes : silhouette score, inertie, Davies-Bouldin", "En calculant l'accuracy", "En comptant les clusters"],
    answer: 1,
    explanation: "Sans labels, on utilise des métriques internes. Silhouette score (cohésion vs séparation), inertie (compacité), Davies-Bouldin (ratio distance intra/inter cluster). Ce sont des proxys, pas des vérités absolues." },

  { id: 114, category: "Non-supervisé", difficulty: "facile",
    question: "Après un K-Means, tu obtiens 3 clusters. Quelle est l'étape suivante ?",
    choices: ["Rien, c'est terminé", "Analyser le profil de chaque cluster pour leur donner un sens métier", "Réentraîner avec plus de K", "Supprimer le plus petit cluster"],
    answer: 1,
    explanation: "Le clustering ne donne que des numéros (0, 1, 2). C'est à toi de regarder les moyennes de chaque feature par cluster et d'interpréter : cluster 0 = jeunes urbains premium, cluster 1 = familles rurales économiques..." },

  { id: 115, category: "Non-supervisé", difficulty: "difficile",
    question: "L'algorithme K-Means est-il garanti de converger ?",
    choices: ["Non, il peut boucler infiniment", "Oui, l'inertie décroît à chaque itération — il converge vers un minimum LOCAL", "Oui, vers le minimum global", "Non, sauf avec k-means++"],
    answer: 1,
    explanation: "À chaque itération, la réassignation + le recalcul des centroïdes ne peut que réduire l'inertie. Mais le résultat dépend de l'initialisation → c'est un minimum local. D'où n_init=10 pour tester plusieurs départs." },

  # ── NON-SUPERVISÉ : PCA (8 questions) ──

  { id: 116, category: "Non-supervisé", difficulty: "facile",
    question: "Que fait la PCA (Principal Component Analysis) ?",
    choices: ["Classifie les données", "Réduit le nombre de dimensions en gardant le maximum de variance", "Regroupe en clusters", "Supprime les outliers"],
    answer: 1,
    explanation: "PCA projette les données de N dimensions vers P dimensions (P < N) en trouvant les axes de variance maximale. 50 features → 2 ou 3 composantes principales pour visualiser ou accélérer." },

  { id: 117, category: "Non-supervisé", difficulty: "moyen",
    question: "Après PCA, la composante PC1 explique 65% de la variance et PC2 explique 20%. Que garder ?",
    choices: ["Uniquement PC1", "PC1 + PC2 = 85% de la variance expliquée — souvent suffisant", "Toutes les composantes", "Aucune, 85% est trop peu"],
    answer: 1,
    explanation: "85% de variance expliquée avec seulement 2 composantes (au lieu de 50 features) est un excellent compromis. En général, on cherche ≥ 80-95% de variance cumulée." },

  { id: 118, category: "Non-supervisé", difficulty: "moyen",
    question: "Pourquoi faut-il OBLIGATOIREMENT scaler avant la PCA ?",
    choices: ["Pour accélérer", "PCA maximise la variance — une feature en km aura plus de variance qu'une en mètres et dominera les composantes", "C'est optionnel", "Pour réduire les NaN"],
    answer: 1,
    explanation: "PCA cherche les directions de variance maximale. Si 'salaire' (en milliers) a plus de variance que 'âge' (en dizaines), PC1 sera 'salaire' déguisé. Le scaling égalise l'importance." },

  { id: 119, category: "Non-supervisé", difficulty: "difficile",
    question: "Les composantes principales sont-elles interprétables ?",
    choices: ["Oui, chaque composante correspond à une feature", "Pas directement — chaque composante est un mélange pondéré de TOUTES les features originales", "Non, elles sont toujours aléatoires", "Oui, grâce aux feature importances"],
    answer: 1,
    explanation: "PC1 = 0.4×surface + 0.3×rooms + 0.2×age + ... Ce sont des combinaisons linéaires. On perd l'interprétabilité directe. On peut regarder les loadings (pca.components_) pour comprendre ce que chaque PC capture." },

  { id: 120, category: "Non-supervisé", difficulty: "facile",
    question: "PCA est souvent utilisée avant un modèle ML. Pourquoi ?",
    choices: ["Elle améliore toujours les résultats", "Elle réduit les dimensions → moins de bruit, entraînement plus rapide", "Elle remplace le modèle", "Elle supprime les outliers"],
    answer: 1,
    explanation: "Moins de features = moins de bruit, moins d'overfitting, entraînement plus rapide. Utile quand il y a beaucoup de features corrélées. Mais attention : on perd en interprétabilité." },

  { id: 121, category: "Non-supervisé", difficulty: "moyen",
    question: "Que montre un <strong>scree plot</strong> (explained variance ratio par composante) ?",
    choices: ["Les corrélations", "Le % de variance expliquée par chaque composante — le 'coude' indique combien garder", "Les clusters", "La loss"],
    answer: 1,
    explanation: "Le scree plot montre la décroissance de variance. Les premières composantes captent beaucoup, puis ça décroît. On cherche le coude où le gain devient marginal. C'est l'Elbow method de la PCA." },

  { id: 122, category: "Non-supervisé", difficulty: "difficile",
    question: "PCA et K-Means peuvent être combinés. Dans quel ordre ?",
    choices: ["K-Means puis PCA", "PCA puis K-Means — on réduit les dimensions d'abord pour améliorer le clustering", "L'ordre n'importe pas", "On ne peut pas les combiner"],
    answer: 1,
    explanation: "En haute dimension, les distances euclidiennes perdent leur sens (malédiction de la dimensionalité). PCA d'abord réduit le bruit et les dimensions, puis K-Means clusterise dans un espace plus propre." },

  { id: 123, category: "Non-supervisé", difficulty: "moyen",
    question: "Que renvoie <code>pca.explained_variance_ratio_</code> ?",
    choices: ["Les composantes principales", "Un array avec le % de variance expliquée par chaque composante", "Les erreurs du modèle", "Les labels des clusters"],
    answer: 1,
    explanation: "Exemple : [0.65, 0.20, 0.08, 0.04, 0.03] → PC1 explique 65%, PC2 20%, etc. La somme cumulée (0.65, 0.85, 0.93...) aide à choisir le nombre de composantes à garder." },

  # ── DEEP LEARNING : NN / MLP (10 questions) ──

  { id: 124, category: "Deep Learning", difficulty: "facile",
    question: "Qu'est-ce qu'un <strong>neurone artificiel</strong> ?",
    choices: ["Un programme autonome", "Un calcul : somme pondérée des entrées + biais, passée dans une fonction d'activation", "Un cluster de données", "Un type de feature"],
    answer: 1,
    explanation: "Neurone = activation(Σ(wᵢ × xᵢ) + b). Chaque entrée est multipliée par un poids, on ajoute un biais, puis on applique une activation (ReLU, sigmoïde...). C'est la brique de base du deep learning." },

  { id: 125, category: "Deep Learning", difficulty: "facile",
    question: "Que fait la fonction d'activation <strong>ReLU</strong> ?",
    choices: ["Normalise entre 0 et 1", "max(0, x) — renvoie 0 si négatif, x si positif", "Renvoie toujours 1", "Calcule la probabilité"],
    answer: 1,
    explanation: "ReLU est simple et efficace : elle 'éteint' les valeurs négatives (→ 0) et laisse passer les positives. C'est la fonction d'activation par défaut dans les couches cachées car elle ne souffre pas du vanishing gradient." },

  { id: 126, category: "Deep Learning", difficulty: "moyen",
    question: "Pour une classification binaire, quelles activation et loss utiliser en sortie ?",
    choices: ["ReLU + MSE", "Sigmoïde + binary_crossentropy", "Softmax + categorical_crossentropy", "Tanh + MAE"],
    answer: 1,
    explanation: "Sigmoïde compresse en [0, 1] → probabilité de la classe positive. Binary crossentropy mesure l'écart entre la probabilité prédite et le label réel (0 ou 1)." },

  { id: 127, category: "Deep Learning", difficulty: "moyen",
    question: "Pour une classification en 5 classes, quelles activation et loss utiliser ?",
    choices: ["Sigmoïde + binary_crossentropy", "Softmax (5 neurones) + categorical_crossentropy", "ReLU + MSE", "Pas d'activation + MAE"],
    answer: 1,
    explanation: "Softmax normalise les 5 sorties pour qu'elles somment à 1 → probabilités par classe. Categorical crossentropy compare cette distribution avec le one-hot du vrai label." },

  { id: 128, category: "Deep Learning", difficulty: "moyen",
    question: "Qu'est-ce que le <strong>backpropagation</strong> ?",
    choices: ["L'ajout de couches", "L'algorithme qui calcule les gradients de la loss par rapport à chaque poids, couche par couche en remontant", "Le chargement des données", "L'initialisation des poids"],
    answer: 1,
    explanation: "Le forward pass calcule la prédiction. Le backward pass (backprop) calcule le gradient de la loss pour chaque poids via la règle de la chaîne. Puis l'optimiseur (Adam, SGD) met à jour les poids." },

  { id: 129, category: "Deep Learning", difficulty: "facile",
    question: "Que fait <code>model.compile(optimizer='adam', loss='mse')</code> ?",
    choices: ["Entraîne le modèle", "Configure l'optimiseur et la fonction de perte AVANT l'entraînement", "Évalue le modèle", "Sauvegarde le modèle"],
    answer: 1,
    explanation: "compile() configure le modèle sans l'entraîner : quel optimiseur (Adam), quelle loss (MSE), quelles métriques (accuracy). C'est un prérequis avant .fit()." },

  { id: 130, category: "Deep Learning", difficulty: "moyen",
    question: "Un modèle a un train loss de 0.02 et un val loss de 0.45 après 100 epochs. Diagnostic ?",
    choices: ["Modèle parfait", "Overfitting sévère — le modèle mémorise le train sans généraliser", "Underfitting", "Les données sont mauvaises"],
    answer: 1,
    explanation: "Grand écart train/val = overfitting classique. Solutions : early stopping, dropout, réduire le nombre de couches/neurones, augmenter les données, ajouter de la régularisation." },

  { id: 131, category: "Deep Learning", difficulty: "moyen",
    question: "Que fait la couche <code>Dropout(0.3)</code> ?",
    choices: ["Supprime 30% des données", "Éteint aléatoirement 30% des neurones à chaque batch pendant l'entraînement", "Réduit le learning rate de 30%", "Supprime 30% des features"],
    answer: 1,
    explanation: "Dropout force le réseau à ne pas dépendre d'un seul neurone. À chaque batch, 30% des neurones sont mis à 0 aléatoirement. Le réseau apprend des représentations plus robustes. Désactivé automatiquement en prédiction." },

  { id: 132, category: "Deep Learning", difficulty: "difficile",
    question: "Pourquoi l'optimiseur <strong>Adam</strong> est-il préféré à SGD ?",
    choices: ["Il est plus simple", "Il adapte le learning rate individuellement pour chaque poids en utilisant les moments (moyenne et variance des gradients)", "Il converge toujours au minimum global", "Il utilise moins de mémoire"],
    answer: 1,
    explanation: "Adam combine Momentum (moyenne mobile des gradients → direction) et RMSProp (variance des gradients → échelle). Chaque poids a son propre learning rate adaptatif. Souvent meilleur que SGD pur, surtout en début d'entraînement." },

  { id: 133, category: "Deep Learning", difficulty: "difficile",
    question: "Qu'est-ce que le <strong>vanishing gradient</strong> et pourquoi ReLU le résout ?",
    choices: ["Un problème de mémoire", "Avec sigmoïde/tanh, les gradients deviennent quasi-nuls dans les couches profondes → les poids ne se mettent plus à jour. ReLU a un gradient constant de 1 pour x > 0", "Un problème de données", "Un type d'overfitting"],
    answer: 1,
    explanation: "Sigmoïde sature à 0 ou 1 → gradient ~0 → les premières couches n'apprennent plus. ReLU : gradient = 1 si x > 0, gradient = 0 si x < 0. Le signal se propage sans s'évanouir (sauf pour les neurones 'morts' à 0)." },

  # ── DEEP LEARNING : CNN (8 questions) ──

  { id: 134, category: "Deep Learning", difficulty: "facile",
    question: "Pourquoi utilise-t-on des CNN plutôt que des réseaux denses pour les images ?",
    choices: ["Les CNN sont plus rapides", "Une image 100×100 = 10 000 features en dense → trop de paramètres. Les convolutions exploitent la structure spatiale locale", "Les CNN sont plus simples", "Les réseaux denses ne gèrent pas les images"],
    answer: 1,
    explanation: "Un réseau dense connecte chaque pixel à chaque neurone → explosion de paramètres. Un CNN utilise des filtres locaux (3×3) qui partagent leurs poids → beaucoup moins de paramètres et capte les patterns locaux (bords, textures)." },

  { id: 135, category: "Deep Learning", difficulty: "moyen",
    question: "Que fait une couche de <strong>convolution</strong> ?",
    choices: ["Réduit la taille de l'image", "Applique des filtres (kernels) qui détectent des patterns locaux : bords, textures, formes", "Aplatit l'image en vecteur", "Classifie l'image"],
    answer: 1,
    explanation: "Un filtre 3×3 glisse sur l'image et calcule un produit scalaire à chaque position. Chaque filtre détecte un pattern différent. La première couche détecte des bords, les couches profondes des formes complexes." },

  { id: 136, category: "Deep Learning", difficulty: "moyen",
    question: "Que fait la couche <code>MaxPooling2D(2, 2)</code> ?",
    choices: ["Ajoute des pixels", "Réduit la taille par 2 en gardant la valeur max dans chaque fenêtre 2×2", "Normalise les pixels", "Augmente le nombre de filtres"],
    answer: 1,
    explanation: "MaxPool(2,2) découpe en fenêtres 2×2 et garde le max de chaque fenêtre. L'image passe de 32×32 à 16×16. Effet : réduit les paramètres, rend le modèle plus robuste aux petits déplacements." },

  { id: 137, category: "Deep Learning", difficulty: "moyen",
    question: "Un CNN typique a l'architecture : Conv→Pool→Conv→Pool→Flatten→Dense→Output. Pourquoi Flatten ?",
    choices: ["Pour accélérer", "Les couches denses attendant un vecteur 1D, pas un tensor 3D — Flatten convertit (7, 7, 64) → (3136,)", "Pour augmenter les features", "C'est optionnel"],
    answer: 1,
    explanation: "Après les convolutions/pooling, les données sont un tensor 3D (hauteur × largeur × filtres). La couche Dense finale nécessite un vecteur 1D pour faire la classification. Flatten aplatit le tensor." },

  { id: 138, category: "Deep Learning", difficulty: "facile",
    question: "Pourquoi normaliser les pixels entre 0 et 1 pour un CNN ?",
    choices: ["C'est obligatoire techniquement", "Des valeurs [0, 255] causent des gradients instables — diviser par 255 stabilise l'entraînement", "Pour réduire la taille du fichier", "Par convention esthétique"],
    answer: 1,
    explanation: "Pixels bruts [0, 255] → valeurs très grandes → gradients instables. Diviser par 255.0 ramène tout entre [0, 1]. L'entraînement converge plus vite et plus stablement." },

  { id: 139, category: "Deep Learning", difficulty: "difficile",
    question: "Qu'est-ce que le <strong>transfer learning</strong> ?",
    choices: ["Transférer les données entre machines", "Utiliser un modèle pré-entraîné (ex: VGG16 sur ImageNet) et l'adapter à sa propre tâche", "Copier les hyperparamètres d'un autre projet", "Entraîner sur plusieurs GPU"],
    answer: 1,
    explanation: "Un modèle entraîné sur ImageNet (1.2M images) a déjà appris à détecter bords, textures, formes. On gèle ces couches et on réentraîne seulement les dernières couches dense sur nos données. Résultat : excellent même avec peu de données." },

  { id: 140, category: "Deep Learning", difficulty: "moyen",
    question: "La <strong>data augmentation</strong> pour les images consiste à :",
    choices: ["Télécharger plus d'images", "Appliquer des transformations (rotation, flip, zoom, crop) pour créer des variantes artificielles", "Supprimer les images floues", "Compresser les images"],
    answer: 1,
    explanation: "Un chat retourné reste un chat. En générant des variantes (flip horizontal, rotation ±15°, zoom), on multiplie artificiellement le dataset → moins d'overfitting, meilleure généralisation." },

  { id: 141, category: "Deep Learning", difficulty: "difficile",
    question: "Quelle est la différence entre <code>padding='same'</code> et <code>padding='valid'</code> en Conv2D ?",
    choices: ["Aucune différence", "'same' ajoute des zéros pour garder la même taille en sortie ; 'valid' réduit la taille", "'valid' est plus rapide", "'same' augmente la résolution"],
    answer: 1,
    explanation: "Un filtre 3×3 sur une image 32×32 avec padding='valid' donne 30×30 (perte de bord). Avec padding='same', on ajoute des zéros autour → sortie 32×32. 'same' est plus courant car il évite la réduction progressive." },

  # ── DEEP LEARNING : RNN / Séquences (6 questions) ──

  { id: 142, category: "Deep Learning", difficulty: "facile",
    question: "Pourquoi utiliser un RNN plutôt qu'un réseau dense pour des séries temporelles ?",
    choices: ["Les RNN sont plus rapides", "Les RNN tiennent compte de l'ORDRE des données — un dense traite chaque timestep indépendamment", "Les RNN ont moins de paramètres", "Le dense ne gère pas les nombres"],
    answer: 1,
    explanation: "La température de demain dépend de celle d'hier et avant-hier. Un réseau dense ignore cet ordre. Le RNN traite les données séquentiellement et maintient un 'état caché' qui encode l'historique." },

  { id: 143, category: "Deep Learning", difficulty: "moyen",
    question: "Quelle est la différence entre un RNN simple et un <strong>LSTM</strong> ?",
    choices: ["Aucune", "Le LSTM a des 'portes' (forget, input, output) qui contrôlent quoi retenir/oublier → mémoire longue", "Le LSTM est plus rapide", "Le RNN simple est plus moderne"],
    answer: 1,
    explanation: "Le RNN simple souffre du vanishing gradient sur les longues séquences. Le LSTM (Long Short-Term Memory) a une cellule de mémoire avec 3 portes qui régulent le flux d'information → il peut retenir des dépendances sur des centaines de timesteps." },

  { id: 144, category: "Deep Learning", difficulty: "moyen",
    question: "Pour prédire le prix d'une action à J+1 avec les 30 derniers jours, quelle shape d'entrée ?",
    choices: ["(30,) — un vecteur de 30 valeurs", "(batch_size, 30, n_features) — 3D : échantillons × timesteps × features", "(30, 30) — matrice carrée", "(1, 30) — une ligne"],
    answer: 1,
    explanation: "Les RNN/LSTM attendent un tensor 3D : (batch_size, sequence_length, n_features). Ici : séquence de 30 jours, avec pour chaque jour les features (open, high, low, close, volume...)." },

  { id: 145, category: "Deep Learning", difficulty: "difficile",
    question: "Qu'est-ce qu'un <strong>GRU</strong> et quand le préférer au LSTM ?",
    choices: ["C'est la même chose", "Le GRU est un LSTM simplifié avec 2 portes au lieu de 3 — plus rapide, performances similaires sur des séquences courtes", "Le GRU est plus ancien", "Le GRU est pour les images"],
    answer: 1,
    explanation: "GRU fusionne la forget gate et l'input gate en une seule 'update gate'. Moins de paramètres → plus rapide à entraîner. Pour des séquences < 100 timesteps, GRU ≈ LSTM. Pour des séquences très longues, LSTM est souvent meilleur." },

  { id: 146, category: "Deep Learning", difficulty: "moyen",
    question: "Pourquoi faut-il créer des <strong>fenêtres glissantes</strong> pour entraîner un RNN sur une série temporelle ?",
    choices: ["Par convention", "Le RNN a besoin de paires (X=fenêtre passée, y=valeur future) — on découpe la série en séquences de longueur fixe", "Pour mélanger les données", "Pour supprimer les outliers"],
    answer: 1,
    explanation: "Série : [10, 12, 11, 15, 14, 16]. Fenêtre de 3 → X=[10,12,11] y=15 ; X=[12,11,15] y=14 ; X=[11,15,14] y=16. On crée des paires input/output pour le supervisé." },

  { id: 147, category: "Deep Learning", difficulty: "difficile",
    question: "Que fait <code>return_sequences=True</code> dans une couche LSTM ?",
    choices: ["Renvoie la prédiction finale", "Renvoie l'état caché à CHAQUE timestep au lieu de seulement le dernier — nécessaire si on empile des LSTM", "Active le dropout", "Renvoie les poids"],
    answer: 1,
    explanation: "Par défaut, LSTM ne renvoie que le dernier état. Avec return_sequences=True, on obtient un état par timestep → output shape (batch, timesteps, units). C'est obligatoire avant une autre couche LSTM." },

  # ── PIPELINE SKLEARN (8 questions) ──

  { id: 148, category: "Pipeline", difficulty: "facile",
    question: "Quel est le principal avantage d'un <code>Pipeline</code> sklearn ?",
    choices: ["Il est plus rapide", "Il élimine le data leakage automatiquement et rend le code reproductible en une seule ligne fit/predict", "Il remplace les modèles", "Il améliore toujours les résultats"],
    answer: 1,
    explanation: "Sans Pipeline, il faut manuellement faire scaler.fit(train) → scaler.transform(test). Risque d'erreur → data leakage. Le Pipeline enchaîne tout : pipe.fit(X_train, y_train) fait fit_transform sur train, transform sur test automatiquement." },

  { id: 149, category: "Pipeline", difficulty: "moyen",
    question: "Que fait <code>ColumnTransformer</code> dans un Pipeline ?",
    choices: ["Supprime des colonnes", "Applique des transformations DIFFÉRENTES selon les colonnes : scaler pour les numériques, encoder pour les catégorielles", "Renomme les colonnes", "Ajoute des colonnes"],
    answer: 1,
    explanation: "ColumnTransformer route les colonnes : les numériques passent par imputer+scaler, les catégorielles par imputer+encoder. Le tout en parallèle. Fini le code spaghetti." },

  { id: 150, category: "Pipeline", difficulty: "moyen",
    question: "Que signifie <code>model__n_estimators</code> dans les paramètres de GridSearchCV ?",
    choices: ["Deux underscores par hasard", "Notation Pipeline : 'model' est le nom de l'étape, 'n_estimators' est le paramètre de cette étape", "C'est une erreur de syntaxe", "model et n_estimators sont indépendants"],
    answer: 1,
    explanation: "Dans Pipeline([('preprocessing', ...), ('model', RandomForest())]), le double underscore navigue : model__n_estimators = le paramètre n_estimators de l'étape nommée 'model'." },

  { id: 151, category: "Pipeline", difficulty: "facile",
    question: "Que fait <code>joblib.dump(pipe, 'pipeline.joblib')</code> ?",
    choices: ["Affiche le pipeline", "Sauvegarde TOUT le pipeline (preprocessing + modèle) dans un fichier binaire", "Supprime le pipeline", "Entraîne le pipeline"],
    answer: 1,
    explanation: "joblib sérialise l'objet Pipeline complet : le scaler (avec ses moyennes/std), l'encoder (avec ses catégories), le modèle (avec ses poids). En production : joblib.load() + pipe.predict()." },

  { id: 152, category: "Pipeline", difficulty: "moyen",
    question: "Peut-on faire un GridSearchCV directement sur un Pipeline ?",
    choices: ["Non, il faut séparer", "Oui — GridSearchCV(pipe, params, cv=5) teste toutes les combinaisons en respectant le pipeline", "Seulement sur le modèle", "Seulement sur le preprocessing"],
    answer: 1,
    explanation: "GridSearchCV respecte le Pipeline : à chaque fold, le preprocessing est fit sur le train et transform sur le val. Pas de data leakage. On peut même tester des paramètres du preprocessing." },

  { id: 153, category: "Pipeline", difficulty: "difficile",
    question: "Pourquoi un Pipeline élimine-t-il le <strong>data leakage</strong> ?",
    choices: ["Il supprime les NaN", "Le fit() est fait UNIQUEMENT sur X_train — le transform du X_test utilise les paramètres appris sur le train", "Il mélange les données", "Il utilise plus de mémoire"],
    answer: 1,
    explanation: "Sans Pipeline, erreur classique : scaler.fit_transform(X) sur tout le dataset, PUIS split. Le scaler a 'vu' le test. Avec Pipeline, le fit() interne ne voit que X_train. C'est la garantie structurelle contre le leakage." },

  { id: 154, category: "Pipeline", difficulty: "moyen",
    question: "Tu veux tester 2 scalers et 3 modèles dans un Pipeline. Combien de combinaisons ?",
    choices: ["5", "6 — 2 × 3 = toutes les combinaisons scaler × modèle", "2", "3"],
    answer: 1,
    explanation: "GridSearchCV teste toutes les combinaisons : (StandardScaler + RF, StandardScaler + XGBoost, StandardScaler + SVM, MinMaxScaler + RF, MinMaxScaler + XGBoost, MinMaxScaler + SVM) = 6 combinaisons, chacune cross-validée." },

  { id: 155, category: "Pipeline", difficulty: "difficile",
    question: "Que fait <code>handle_unknown='ignore'</code> dans OneHotEncoder au sein d'un Pipeline ?",
    choices: ["Supprime les colonnes inconnues", "Si le test contient une catégorie jamais vue dans le train, elle est encodée en vecteur de zéros au lieu de lever une erreur", "Ignore le OneHotEncoding", "Supprime les lignes inconnues"],
    answer: 1,
    explanation: "En production, un nouvel utilisateur peut venir d'une ville jamais vue dans le train. Sans handle_unknown='ignore', le code plante. Avec, la ville est encodée [0, 0, ..., 0] → le modèle fait de son mieux." },

  # ── MLOPS (14 questions) ──

  { id: 156, category: "MLOps", difficulty: "facile",
    question: "Qu'est-ce qu'une <strong>API</strong> dans le contexte MLOps ?",
    choices: ["Un type de modèle", "Un serveur qui reçoit des requêtes HTTP et renvoie des prédictions en JSON", "Un format de données", "Un type de base de données"],
    answer: 1,
    explanation: "L'API est l'interface entre le monde extérieur et ton modèle. Tu envoies les features par HTTP, le serveur charge le modèle, fait la prédiction, et renvoie le résultat en JSON." },

  { id: 157, category: "MLOps", difficulty: "facile",
    question: "Que fait FastAPI par rapport à Flask dans le cadre du Wagon ?",
    choices: ["C'est la même chose", "FastAPI type automatiquement les paramètres, génère la doc Swagger, et est asynchrone par défaut", "Flask est meilleur en production", "FastAPI est seulement pour le frontend"],
    answer: 1,
    explanation: "FastAPI : typage (age: int vérifié automatiquement), doc interactive (/docs), validation, performances async. Flask est plus simple mais il faut tout faire manuellement." },

  { id: 158, category: "MLOps", difficulty: "moyen",
    question: "Pourquoi utiliser <code>app.state.model</code> au lieu de charger le modèle dans la fonction predict() ?",
    choices: ["C'est plus joli", "Le modèle est chargé UNE SEULE FOIS au démarrage au lieu de à chaque requête → 10× plus rapide", "app.state est obligatoire", "Ça ne change rien"],
    answer: 1,
    explanation: "joblib.load() peut prendre 2 secondes. Si tu le mets dans predict(), chaque requête attend 2s. Avec app.state, le load se fait au démarrage et l'objet reste en mémoire. Critique pour les modèles Deep Learning." },

  { id: 159, category: "MLOps", difficulty: "moyen",
    question: "Que fait la commande <code>uvicorn api:app --reload</code> ?",
    choices: ["Entraîne le modèle", "Lance le serveur FastAPI avec rechargement automatique à chaque modification du code", "Déploie sur le cloud", "Teste les endpoints"],
    answer: 1,
    explanation: "uvicorn = serveur ASGI. api:app → fichier api.py, objet app (FastAPI). --reload relance le serveur à chaque sauvegarde du code. Ne PAS utiliser --reload en production (performance)." },

  { id: 160, category: "MLOps", difficulty: "facile",
    question: "Quel endpoint génère automatiquement la documentation interactive de FastAPI ?",
    choices: ["/api", "/docs — page Swagger générée automatiquement", "/admin", "/test"],
    answer: 1,
    explanation: "FastAPI crée automatiquement /docs (Swagger UI) et /redoc (documentation alternative). On peut y tester chaque endpoint avec le bouton 'Try it out'. Pas besoin de Postman." },

  { id: 161, category: "MLOps", difficulty: "moyen",
    question: "Qu'est-ce que <strong>Docker</strong> fait concrètement ?",
    choices: ["C'est un cloud provider", "Il empaquette code + dépendances + OS dans un conteneur isolé — garanti identique partout", "Il optimise le modèle", "C'est un outil de versioning"],
    answer: 1,
    explanation: "'Ça marche sur ma machine' → Docker résout ça. Le conteneur embarque Python, les packages, le modèle, le code. Ce qui marche en local marchera exactement pareil sur Cloud Run, sur le PC du collègue, etc." },

  { id: 162, category: "MLOps", difficulty: "moyen",
    question: "Dans le Dockerfile, pourquoi copier requirements.txt AVANT le code ?",
    choices: ["Par convention", "Pour profiter du cache Docker — si les dépendances n'ont pas changé, Docker skip l'install et va plus vite", "C'est obligatoire", "Pour des raisons de sécurité"],
    answer: 1,
    explanation: "Docker construit en couches. Si la couche 'COPY requirements.txt' n'a pas changé depuis le dernier build, Docker réutilise le cache. Le 'pip install' (lent) est skipé. Seul le code (qui change souvent) est recopié." },

  { id: 163, category: "MLOps", difficulty: "facile",
    question: "Pourquoi utiliser <code>python:3.10-slim</code> plutôt que <code>python:3.10</code> dans le Dockerfile ?",
    choices: ["Aucune différence", "L'image slim fait ~150 MB au lieu de ~900 MB — plus rapide à build et à déployer", "slim est plus récent", "slim a plus de packages"],
    answer: 1,
    explanation: "L'image 'complète' inclut des outils système inutiles en production. 'slim' contient juste le nécessaire pour faire tourner Python. Moins de taille = build plus rapide, déploiement plus rapide, moins de surface d'attaque." },

  { id: 164, category: "MLOps", difficulty: "moyen",
    question: "Que fait la variable <code>$PORT</code> dans <code>CMD uvicorn api:app --port $PORT</code> ?",
    choices: ["Fixe toujours le port à 8080", "Cloud Run injecte dynamiquement le port — le code doit le lire via cette variable", "C'est le port de la base de données", "C'est optionnel"],
    answer: 1,
    explanation: "Cloud Run attribue un port dynamiquement et le passe via la variable d'environnement PORT. Si tu hardcodes 8080, l'app ne démarre pas. --port $PORT s'adapte automatiquement." },

  { id: 165, category: "MLOps", difficulty: "moyen",
    question: "Que fait <code>gcloud builds submit --tag gcr.io/PROJECT/my-model</code> ?",
    choices: ["Lance l'API en local", "Build l'image Docker dans le cloud GCP et la pousse sur le Container Registry", "Entraîne le modèle sur GCP", "Supprime le projet"],
    answer: 1,
    explanation: "Pas besoin d'avoir Docker installé en local. GCP build l'image dans le cloud à partir du Dockerfile + code, puis la stocke sur le Container Registry (gcr.io). Prêt pour Cloud Run." },

  { id: 166, category: "MLOps", difficulty: "facile",
    question: "Qu'est-ce que <strong>Cloud Run</strong> ?",
    choices: ["Un GPU cloud", "Un service serverless qui lance ton conteneur Docker et scale automatiquement (0 à N instances)", "Un service de stockage", "Un IDE en ligne"],
    answer: 1,
    explanation: "Cloud Run = 'donne-moi une image Docker, je m'occupe du reste'. 0 requête = 0 instance (0€). 1000 req/sec = auto-scaling. HTTPS automatique. Tu ne gères aucun serveur." },

  { id: 167, category: "MLOps", difficulty: "moyen",
    question: "Pourquoi ajouter <code>--allow-unauthenticated</code> au déploiement Cloud Run ?",
    choices: ["C'est obligatoire", "Pour rendre l'API accessible publiquement sans authentification", "Pour accélérer le déploiement", "Pour réduire les coûts"],
    answer: 1,
    explanation: "Par défaut, Cloud Run exige une authentification IAM. --allow-unauthenticated permet à n'importe qui d'appeler l'API (utile pour un TP ou un MVP). En production, on protège avec des tokens." },

  { id: 168, category: "MLOps", difficulty: "difficile",
    question: "Pourquoi stocker le modèle sur Google Cloud Storage (GCS) plutôt que dans l'image Docker ?",
    choices: ["GCS est plus rapide", "Un modèle lourd (> 500 MB) gonfle l'image Docker → build lent, déploiement lent. GCS permet de le charger séparément au démarrage", "Docker ne supporte pas les gros fichiers", "C'est une question de sécurité"],
    answer: 1,
    explanation: "Un modèle Deep Learning de 2 Go dans le Docker = image de 2 Go à build et deployer à chaque changement de code. Mieux : stocker le modèle sur GCS et le télécharger au démarrage (app.state)." },

  { id: 169, category: "MLOps", difficulty: "difficile",
    question: "Tu modifies une feature dans le preprocessing. Que faut-il redéployer ?",
    choices: ["Seulement le modèle", "TOUT : réentraîner le modèle avec la nouvelle feature, re-sauvegarder le pipeline, rebuild l'image Docker, redéployer", "Seulement l'API", "Rien, c'est automatique"],
    answer: 1,
    explanation: "Le pipeline contient le preprocessing + le modèle ensemble. Si tu changes le preprocessing (nouvelle feature, nouveau scaler), il faut réentraîner, re-sérialiser (joblib.dump), rebuild Docker, redéployer. C'est la chaîne complète." },

  # ── MÉTHODES ML / TRANSVERSAL (15 questions) ──

  { id: 170, category: "Méthodes ML", difficulty: "facile",
    question: "Quelle est la différence entre <strong>classification</strong> et <strong>régression</strong> ?",
    choices: ["Aucune", "Classification prédit une catégorie (spam/pas spam) ; régression prédit un nombre continu (prix en €)", "La régression est plus rapide", "La classification n'utilise que des arbres"],
    answer: 1,
    explanation: "Classification : la target est discrète (chat/chien, 0/1). Régression : la target est continue (prix, température). Le choix du problème détermine le modèle et les métriques." },

  { id: 171, category: "Méthodes ML", difficulty: "moyen",
    question: "Qu'est-ce que la <strong>malédiction de la dimensionalité</strong> ?",
    choices: ["Trop de lignes", "En haute dimension, les distances deviennent peu informatives — les points sont tous 'loin' les uns des autres", "Trop de catégories", "Des données trop petites"],
    answer: 1,
    explanation: "Avec 1000 features, l'espace est si vaste que tous les points sont quasi-équidistants. KNN, K-Means et toute méthode basée sur les distances souffrent. Solutions : PCA, feature selection, ou modèles basés sur les arbres." },

  { id: 172, category: "Méthodes ML", difficulty: "facile",
    question: "Qu'est-ce que l'<strong>overfitting</strong> ?",
    choices: ["Le modèle est trop simple", "Le modèle mémorise le bruit du train set et ne généralise pas sur de nouvelles données", "Le modèle met trop de temps", "Les données sont trop propres"],
    answer: 1,
    explanation: "Overfitting = bon sur le train, mauvais sur le test. Le modèle a appris les particularités du train (bruit, outliers) au lieu des patterns généraux. Solutions : régularisation, cross-validation, early stopping, plus de données." },

  { id: 173, category: "Méthodes ML", difficulty: "moyen",
    question: "Qu'est-ce que le <strong>compromis biais-variance</strong> ?",
    choices: ["Choisir entre vitesse et précision", "Un modèle simple a un biais fort (underfitting) ; un modèle complexe a une variance forte (overfitting). Il faut trouver le juste milieu", "C'est la même chose que le data leakage", "Ça ne concerne que le deep learning"],
    answer: 1,
    explanation: "Biais fort = le modèle est trop simple, il rate le pattern (ex: droite pour une courbe). Variance forte = trop complexe, instable d'un dataset à l'autre. L'objectif est de minimiser l'erreur totale = biais² + variance." },

  { id: 174, category: "Méthodes ML", difficulty: "moyen",
    question: "Tu as 500 lignes et 200 features. Quel risque principal ?",
    choices: ["Underfitting", "Overfitting — trop de features pour si peu de données (le modèle peut 'tricher')", "Données trop propres", "Calcul trop long"],
    answer: 1,
    explanation: "Plus de features que de lignes → le modèle peut trouver des patterns accidentels dans le bruit. Solutions : feature selection, PCA pour réduire à ~20 composantes, régularisation forte (Lasso, Ridge)." },

  { id: 175, category: "Méthodes ML", difficulty: "facile",
    question: "Le <code>random_state=42</code> sert à quoi ?",
    choices: ["Améliorer les résultats", "Fixer la graine aléatoire pour obtenir des résultats reproductibles", "Utiliser 42 features", "Limiter à 42 itérations"],
    answer: 1,
    explanation: "Beaucoup d'algorithmes utilisent l'aléatoire (split, bootstrap, initialisation). random_state fixe le générateur → même code = même résultat. 42 par convention (Hitchhiker's Guide), n'importe quel entier fonctionne." },

  { id: 176, category: "Méthodes ML", difficulty: "moyen",
    question: "Qu'est-ce que le <strong>feature engineering</strong> ?",
    choices: ["Supprimer les features inutiles", "Créer de nouvelles features à partir des existantes pour donner plus d'information au modèle", "Renommer les colonnes", "Normaliser les données"],
    answer: 1,
    explanation: "Exemples : surface_par_pièce = surface/rooms, est_weekend = jour in [samedi, dimanche], ancienneté = 2025 − année_construction. Le feature engineering est souvent ce qui fait la plus grande différence de performance." },

  { id: 177, category: "Méthodes ML", difficulty: "difficile",
    question: "Pourquoi la <strong>target encoding</strong> peut-elle provoquer du data leakage ?",
    choices: ["Elle est trop lente", "Elle utilise la target (y) pour encoder X — si mal fait, le modèle voit indirectement la réponse", "Elle crée trop de colonnes", "C'est faux, aucun risque"],
    answer: 1,
    explanation: "Target encoding remplace chaque catégorie par la moyenne de y pour cette catégorie. Si calculé sur tout le dataset, le modèle 'voit' y dans X. Solution : calculer uniquement sur le train, ou utiliser le smoothing et le fold encoding." },

  { id: 178, category: "Méthodes ML", difficulty: "moyen",
    question: "Quand utiliser l'<strong>AUC-ROC</strong> comme métrique ?",
    choices: ["Toujours en régression", "En classification binaire — elle mesure la capacité du modèle à distinguer les deux classes, indépendamment du seuil", "Seulement pour le deep learning", "Quand il y a plus de 3 classes"],
    answer: 1,
    explanation: "L'AUC-ROC trace le True Positive Rate vs False Positive Rate pour tous les seuils. AUC = 0.5 → modèle aléatoire. AUC = 1 → parfait. C'est utile quand le seuil optimal n'est pas encore choisi." },

  { id: 179, category: "Méthodes ML", difficulty: "facile",
    question: "Que signifie <code>n_jobs=-1</code> dans sklearn ?",
    choices: ["Aucun job en parallèle", "Utilise TOUS les cœurs CPU disponibles pour paralléliser le calcul", "Limite à 1 cœur", "Désactive le calcul"],
    answer: 1,
    explanation: "n_jobs=-1 dans GridSearchCV, cross_val_score, ou RandomForest utilise tous les cœurs du processeur. Si tu as 8 cœurs, 8 folds de cross-validation tournent en même temps. Speedup quasi-linéaire." },

  { id: 180, category: "Méthodes ML", difficulty: "moyen",
    question: "Quelle est la différence entre <code>GridSearchCV</code> et <code>RandomizedSearchCV</code> ?",
    choices: ["Aucune", "Grid teste TOUTES les combinaisons ; Randomized en teste un échantillon aléatoire → plus rapide quand l'espace est grand", "Randomized est plus précis", "Grid est pour la régression"],
    answer: 1,
    explanation: "3 paramètres × 10 valeurs = 1000 combinaisons × 5 folds = 5000 fits. GridSearch les fait tous. RandomizedSearchCV(n_iter=50) n'en teste que 50 → 100× plus rapide, souvent presque aussi bon." },

  { id: 181, category: "Méthodes ML", difficulty: "difficile",
    question: "Qu'est-ce que la <strong>stratification</strong> dans train_test_split ?",
    choices: ["Trier les données", "Garantir que chaque split a les mêmes proportions de classes que le dataset original", "Normaliser les données", "Supprimer les doublons"],
    answer: 1,
    explanation: "Si tu as 5% de classe 1, un split aléatoire pourrait mettre 0% de classe 1 dans le test. stratify=y force les mêmes proportions (5%) dans train et test. Critique pour les classes déséquilibrées." },

  { id: 182, category: "Méthodes ML", difficulty: "moyen",
    question: "Accuracy = 97% sur un dataset avec 97% de classe 0. Ce modèle est-il bon ?",
    choices: ["Oui, 97% c'est excellent", "Non — un modèle qui prédit toujours 0 a aussi 97%. Il faut regarder recall, precision, F1 sur la classe 1", "Ça dépend du modèle", "Il faut plus de données"],
    answer: 1,
    explanation: "L'accuracy paradox : sur des classes déséquilibrées, l'accuracy est trompeuse. Le modèle ignore peut-être complètement la classe rare. Toujours vérifier le classification_report et la confusion matrix." },

  { id: 183, category: "Méthodes ML", difficulty: "difficile",
    question: "Quel est l'effet de l'<strong>early stopping</strong> en deep learning ?",
    choices: ["Arrêter l'entraînement à une heure fixe", "Stopper quand la val_loss ne s'améliore plus depuis N epochs → éviter l'overfitting", "Arrêter quand le train_loss atteint 0", "Limiter le nombre de couches"],
    answer: 1,
    explanation: "On surveille la val_loss. Si elle remonte pendant 'patience' epochs (ex: 10), on arrête et on restaure les meilleurs poids. C'est la régularisation la plus simple et efficace en deep learning." },

  { id: 184, category: "Méthodes ML", difficulty: "moyen",
    question: "Pourquoi ne PAS mettre les secrets (clés API, mots de passe) dans le code ?",
    choices: ["C'est plus lent", "Si le code est sur Git, les secrets sont exposés publiquement. Utiliser des variables d'environnement (.env)", "Les secrets ne marchent que en local", "C'est une convention sans importance"],
    answer: 1,
    explanation: "Un .env est ignoré par Git (.gitignore). Les variables d'environnement sont injectées au runtime. En production, Cloud Run/Heroku gèrent les secrets séparément. Jamais de mot de passe en dur dans le code." },

  # ── QUESTIONS TRANSVERSALES / SCÉNARIOS (16 questions) ──

  { id: 185, category: "Méthodes ML", difficulty: "moyen",
    question: "Tu veux prédire si un client va se désabonner (churn). Quel type de problème ?",
    choices: ["Régression", "Classification binaire (churn = oui/non)", "Clustering", "Réduction de dimension"],
    answer: 1,
    explanation: "Churn = oui ou non → 2 classes → classification binaire. Modèles adaptés : Logistic Regression, Random Forest, XGBoost. Métriques : recall (ne pas rater de churners), F1, AUC-ROC." },

  { id: 186, category: "Méthodes ML", difficulty: "moyen",
    question: "Tu as un dataset de 50 lignes et 3 features. Quel modèle choisir ?",
    choices: ["Deep Learning avec 10 couches", "Un modèle simple : régression linéaire/logistique ou KNN", "XGBoost avec 1000 arbres", "CNN"],
    answer: 1,
    explanation: "50 lignes = très peu de données. Un modèle complexe va overfitter immédiatement. La régression linéaire/logistique a peu de paramètres et généralise bien même avec peu de données." },

  { id: 187, category: "Méthodes ML", difficulty: "difficile",
    question: "Qu'est-ce qu'un <strong>modèle de base</strong> (baseline) et pourquoi en faut-il un ?",
    choices: ["Le premier modèle qu'on essaie", "Un modèle naïf (prédire la moyenne, la classe majoritaire) qui sert de référence minimum à battre", "Le modèle final en production", "Un modèle sans features"],
    answer: 1,
    explanation: "Avant tout ML, on établit une baseline : en régression → prédire la moyenne ; en classification → prédire la classe majoritaire. Si ton modèle ne bat pas ça, il n'a rien appris d'utile." },

  { id: 188, category: "Preprocessing", difficulty: "moyen",
    question: "Tu vois que 40% de la colonne 'revenue' est manquante. Que faire ?",
    choices: ["Supprimer la colonne sans réfléchir", "Investiguer POURQUOI les données manquent, puis imputer (median) ou créer une feature 'revenue_is_missing'", "Remplir par 0", "Supprimer 40% des lignes"],
    answer: 1,
    explanation: "40% de NaN c'est beaucoup. D'abord comprendre : est-ce aléatoire ou structurel (les non-déclarants) ? Si le fait d'être manquant est informatif, ajouter une colonne binaire 'is_missing'. Puis imputer la médiane pour les valeurs manquantes." },

  { id: 189, category: "Méthodes ML", difficulty: "facile",
    question: "Que fait <code>model.score(X_test, y_test)</code> ?",
    choices: ["Entraîne le modèle", "Renvoie la métrique par défaut : R² en régression, accuracy en classification", "Affiche les paramètres", "Sauvegarde le modèle"],
    answer: 1,
    explanation: "Raccourci sklearn. En régression, score() renvoie le R². En classification, l'accuracy. Pour d'autres métriques (F1, RMSE), utiliser les fonctions de sklearn.metrics." },

  { id: 190, category: "Boosting", difficulty: "moyen",
    question: "Quelle est la fonction de <code>eval_set</code> dans xgb.fit() ?",
    choices: ["Le jeu de test final", "Un jeu de validation pour surveiller la performance et activer l'early stopping", "Les hyperparamètres", "Le jeu d'entraînement"],
    answer: 1,
    explanation: "eval_set=[(X_val, y_val)] permet à XGBoost de calculer la loss sur la validation à chaque itération. Combiné à early_stopping_rounds=10, l'entraînement s'arrête quand la val_loss stagne." },

  { id: 191, category: "Deep Learning", difficulty: "moyen",
    question: "Qu'est-ce qu'une <strong>epoch</strong> ?",
    choices: ["Un batch de données", "Un passage complet sur TOUT le dataset d'entraînement", "Une prédiction", "Un pas de gradient"],
    answer: 1,
    explanation: "1 epoch = le modèle a vu chaque exemple une fois. Avec 1000 exemples et batch_size=100, 1 epoch = 10 batches = 10 mises à jour des poids. On entraîne typiquement 10 à 200 epochs." },

  { id: 192, category: "Deep Learning", difficulty: "moyen",
    question: "Qu'est-ce que le <strong>batch_size</strong> et comment l'ajuster ?",
    choices: ["Le nombre total d'exemples", "Le nombre d'exemples traités avant chaque mise à jour des poids — petit batch = bruit mais bonne généralisation", "Le nombre de couches", "Le nombre d'epochs"],
    answer: 1,
    explanation: "Batch=32 → les poids sont mis à jour après 32 exemples. Petit batch (16, 32) = gradients bruités mais meilleure généralisation. Grand batch (256, 512) = gradients stables mais peut rester bloqué dans des minima." },

  { id: 193, category: "Non-supervisé", difficulty: "moyen",
    question: "Tu appliques K-Means et obtiens des clusters de tailles 450, 380, 15, et 5. Que suspecter ?",
    choices: ["Résultat parfait", "Les clusters 15 et 5 sont probablement des outliers ou un K trop grand", "Il faut augmenter K", "C'est toujours normal"],
    answer: 1,
    explanation: "Des clusters minuscules sont souvent des outliers que K-Means isole. Solutions : essayer un K plus petit, supprimer les outliers d'abord, ou utiliser DBSCAN qui les détecte nativement." },

  { id: 194, category: "Pipeline", difficulty: "moyen",
    question: "Pourquoi le Pipeline est-il important en <strong>production</strong> et pas seulement en exploration ?",
    choices: ["Il est plus rapide", "En production, le même preprocessing doit être appliqué à la donnée brute de l'utilisateur — le Pipeline garantit la cohérence", "Il est joli dans le code", "C'est optionnel en production"],
    answer: 1,
    explanation: "En production, l'utilisateur envoie des données brutes. Le Pipeline fait automatiquement : imputer → scaler (avec les stats du train) → encoder → prédire. Sans Pipeline, il faut tout refaire manuellement → source de bugs." },

  { id: 195, category: "MLOps", difficulty: "moyen",
    question: "Tu changes le code de l'API mais pas le modèle. Que redéployer ?",
    choices: ["Tout depuis l'entraînement", "Seulement rebuild l'image Docker et redéployer — pas besoin de réentraîner le modèle", "Rien", "Seulement le modèle"],
    answer: 1,
    explanation: "Si le modèle (pipeline.joblib) n'a pas changé, on ne réentraîne pas. On rebuild l'image Docker (qui inclut le nouveau code) et on redéploie. Grâce au cache Docker, le pip install est skipé → build rapide." },

  { id: 196, category: "Régression Logistique", difficulty: "difficile",
    question: "La courbe ROC de ton modèle passe par le point (FPR=0.1, TPR=0.9). Que signifie-t-il ?",
    choices: ["Le modèle est mauvais", "Au seuil correspondant, le modèle détecte 90% des positifs en acceptant 10% de faux positifs", "FPR=0.1 signifie 10% d'accuracy", "Le modèle est aléatoire"],
    answer: 1,
    explanation: "TPR=0.9 → on détecte 90% des vrais positifs (recall). FPR=0.1 → on classe à tort 10% des négatifs comme positifs. C'est un excellent point. La diagonale (TPR=FPR) serait un modèle aléatoire." },

  { id: 197, category: "Arbres & Random Forest", difficulty: "moyen",
    question: "Un Random Forest n'a PAS besoin de scaling. Pourquoi ?",
    choices: ["C'est faux, il faut toujours scaler", "Les arbres font des splits (if x > seuil), pas des calculs de distance — l'échelle n'a aucune importance", "Le scaling est intégré", "C'est un bug connu"],
    answer: 1,
    explanation: "Un arbre demande : 'surface > 80 ?' La réponse ne change pas si surface est en m² ou en km². Les arbres (et donc RF, XGBoost) sont invariants au scaling. C'est un de leurs avantages." },

  { id: 198, category: "Deep Learning", difficulty: "difficile",
    question: "Qu'est-ce qu'un <strong>learning rate scheduler</strong> ?",
    choices: ["Un planning d'entraînement", "Un mécanisme qui diminue le learning rate au fil des epochs pour affiner la convergence", "Un remplacement de l'optimiseur", "Un type de couche"],
    answer: 1,
    explanation: "Au début, un LR élevé permet d'avancer vite. Plus on s'approche du minimum, plus il faut des petits pas. Le scheduler réduit le LR automatiquement (ex: diviser par 10 tous les 30 epochs) pour converger plus finement." },

  { id: 199, category: "Méthodes ML", difficulty: "moyen",
    question: "Qu'est-ce que le <strong>No Free Lunch theorem</strong> signifie en pratique ?",
    choices: ["Il faut payer pour les données", "Aucun algorithme n'est universellement le meilleur — il faut tester plusieurs modèles sur chaque problème", "Le deep learning est toujours meilleur", "Les données sont toujours insuffisantes"],
    answer: 1,
    explanation: "Le théorème dit qu'aucun modèle ne domine tous les autres sur tous les problèmes. XGBoost est souvent bon sur du tabulaire, les CNN sur les images, mais il n'y a pas de certitude. D'où l'importance du benchmarking." },

  { id: 200, category: "Méthodes ML", difficulty: "difficile",
    question: "Tu déploies un modèle de prédiction de prix immobilier. 6 mois plus tard, les prédictions se dégradent. Pourquoi ?",
    choices: ["Le serveur est trop lent", "Le <strong>data drift</strong> : le marché a changé, les données de production ne ressemblent plus à celles d'entraînement", "Le modèle a été supprimé", "Le dataset était trop petit"],
    answer: 1,
    explanation: "Data drift = la distribution des données change au fil du temps. Les prix montent, de nouveaux quartiers émergent, les taux d'intérêt changent. Le modèle entraîné sur des données anciennes n'est plus adapté → réentraîner périodiquement." },

)
