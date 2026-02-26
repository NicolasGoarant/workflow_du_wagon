# frozen_string_literal: true
# Quiz questions 104-200 — positions randomisées, longueurs équilibrées

QUIZ_QUESTIONS.push(

  # ── NON-SUPERVISÉ : K-MEANS (12 questions) ──

  { id: 104, category: "Non-supervisé", difficulty: "facile",
    question: "Quelle est la différence fondamentale entre supervisé et non-supervisé ?",
    choices: ["En non-supervisé, il n'y a PAS de target", "Le supervisé est plus rapide", "Le non-supervisé utilise plus de données", "Le supervisé ne marche qu'en classification"],
    answer: 0,
    explanation: "En supervisé on a des étiquettes (y) : spam/pas spam, prix en €. En non-supervisé, on a juste X et on cherche des structures cachées : groupes, axes principaux, anomalies." },

  { id: 105, category: "Non-supervisé", difficulty: "facile",
    question: "Que fait K-Means ?",
    choices: ["Prédit une valeur continue, selon les bonnes pratiques", "Réduit les dimensions, c'est la norme en data science", "Classifie avec des labels connus, dans la plupart des cas", "Regroupe les données en K clusters en minimisant la distance intra-cluster"],
    answer: 3,
    explanation: "K-Means place K centroïdes, assigne chaque point au centroïde le plus proche, puis recalcule les centroïdes. On répète jusqu'à convergence. Résultat : K groupes homogènes." },

  { id: 106, category: "Non-supervisé", difficulty: "moyen",
    question: "Comment choisir le nombre K de clusters ?",
    choices: ["K = nombre de lignes / 100, c'est recommandé par défaut", "La méthode du coude (Elbow)", "Toujours K=3, même en production", "K = nombre de features, c'est la méthode standard"],
    answer: 1,
    explanation: "On teste K=2, 3, 4... et on trace l'inertie. L'inertie baisse toujours, mais il y a un 'coude' où le gain ralentit. Ce coude indique le bon K. On peut compléter avec le silhouette score." },

  { id: 107, category: "Non-supervisé", difficulty: "moyen",
    question: "Qu'est-ce que l'<strong>inertie</strong> en K-Means ?",
    choices: ["Le nombre d'itérations, comme en machine learning classique", "La somme des distances au carré entre chaque point et son centroïde", "La vitesse de convergence, c'est la méthode standard", "Le nombre de clusters, pour tous les modèles"],
    answer: 1,
    explanation: "Inertie = Σ ||xᵢ − centroïde||². Plus l'inertie est basse, plus les clusters sont compacts. Mais inertie = 0 arrive avec K = nombre de points (chaque point est son propre cluster) → pas utile." },

  { id: 108, category: "Non-supervisé", difficulty: "moyen",
    question: "K-Means est sensible à l'initialisation des centroïdes. Comment sklearn résout-il ce problème ?",
    choices: ["Il n'y a pas de solution, pour tous les modèles, bien que ce soit un choix fréquent chez les débutants", "Il utilise toujours les mêmes centroïdes, sans tenir compte des spécificités du problème", "L'option init='k-means++' place les centroïdes intelligemment, puis n_init=10 fait 10 essais", "Il supprime les outliers d'abord, selon les bonnes pratiques"],
    answer: 2,
    explanation: "k-means++ espace les centroïdes initiaux pour éviter qu'ils soient proches. Puis n_init=10 lance K-Means 10 fois avec des initialisations différentes et garde le meilleur résultat (inertie la plus basse)." },

  { id: 109, category: "Non-supervisé", difficulty: "difficile",
    question: "Le <strong>silhouette score</strong> d'un point est de -0.3. Que signifie-t-il ?",
    choices: ["C'est un outlier à supprimer, comme indiqué dans les tutoriels de référence", "Le point est bien placé, dans un contexte ML classique", "Le point est probablement dans le MAUVAIS cluster", "Le score est invalide, dans la majorité des pipelines"],
    answer: 2,
    explanation: "Silhouette = (b − a) / max(a, b) où a = distance moyenne au sein du cluster, b = distance moyenne au cluster le plus proche. Score négatif → a > b → le point est plus proche d'un autre cluster. Score entre -1 (mauvais) et +1 (parfait)." },

  { id: 110, category: "Non-supervisé", difficulty: "facile",
    question: "Pourquoi faut-il scaler les données AVANT K-Means ?",
    choices: ["C'est optionnel, c'est la norme en data science", "Parce que sklearn l'exige, c'est la pratique standard en machine learning supervisé", "K-Means utilise la distance euclidienne", "Pour accélérer le calcul, mais ce n'est pas la meilleure approche pour ce cas"],
    answer: 2,
    explanation: "Si 'revenu' va de 20K à 200K et 'âge' de 18 à 80, les distances seront dominées par le revenu. Le scaling ramène tout à la même échelle pour que chaque feature pèse de façon équitable." },

  { id: 111, category: "Non-supervisé", difficulty: "moyen",
    question: "K-Means produit toujours des clusters de forme sphérique. Quand est-ce un problème ?",
    choices: ["Jamais, selon la documentation sklearn, quelle que soit la taille du dataset", "Quand les vrais groupes ont des formes allongées, en anneaux ou de tailles très différentes", "Seulement en 3D, dans un contexte ML classique, comme indiqué dans les tutoriels de référence", "Quand K est pair, c'est la méthode standard, même si certains praticiens le font en exploration"],
    answer: 1,
    explanation: "K-Means utilise la distance au centroïde → clusters convexes et sphériques. Pour des formes irrégulières, utiliser DBSCAN (basé sur la densité) ou Gaussian Mixture Models." },

  { id: 112, category: "Non-supervisé", difficulty: "difficile",
    question: "Qu'est-ce que <strong>DBSCAN</strong> et quel avantage a-t-il sur K-Means ?",
    choices: ["Un scaler plus rapide, selon la documentation sklearn, indépendamment du type de modèle utilisé", "Un random forest non-supervisé, pour tous les modèles, selon les principes fondamentaux du machine learning", "Un type de réseau de neurones, selon les bonnes pratiques, d'après les conventions établies en data science", "Un clustering basé sur la densité qui détecte les clusters de forme arbitraire et les outliers, sans choisir K"],
    answer: 3,
    explanation: "DBSCAN regroupe les points denses et marque les points isolés comme bruit (-1). Avantages : pas besoin de choisir K, détecte les outliers, clusters de toute forme. Inconvénient : deux hyperparamètres (eps, min_samples) à régler." },

  { id: 113, category: "Non-supervisé", difficulty: "moyen",
    question: "Comment évaluer un clustering quand on N'A PAS de labels ?",
    choices: ["En comptant les clusters, c'est la méthode standard", "Impossible sans labels, c'est recommandé par défaut", "Avec des métriques internes", "En calculant l'accuracy, pour des raisons de performance"],
    answer: 2,
    explanation: "Sans labels, on utilise des métriques internes. Silhouette score (cohésion vs séparation), inertie (compacité), Davies-Bouldin (ratio distance intra/inter cluster). Ce sont des proxys, pas des vérités absolues." },

  { id: 114, category: "Non-supervisé", difficulty: "facile",
    question: "Après un K-Means, tu obtiens 3 clusters. Quelle est l'étape suivante ?",
    choices: ["Supprimer le plus petit cluster, selon la documentation sklearn", "Réentraîner avec plus de K, dans un contexte ML classique", "Rien, c'est terminé, selon la documentation sklearn", "Analyser le profil de chaque cluster pour leur donner un sens métier"],
    answer: 3,
    explanation: "Le clustering ne donne que des numéros (0, 1, 2). C'est à toi de regarder les moyennes de chaque feature par cluster et d'interpréter : cluster 0 = jeunes urbains premium, cluster 1 = familles rurales économiques..." },

  { id: 115, category: "Non-supervisé", difficulty: "difficile",
    question: "L'algorithme K-Means est-il garanti de converger ?",
    choices: ["Oui, vers le minimum global", "Non, il peut boucler infiniment", "Oui, l'inertie décroît à chaque itération", "Non, sauf avec k-means++, c'est la configuration par défaut de la plupart des frameworks"],
    answer: 2,
    explanation: "À chaque itération, la réassignation + le recalcul des centroïdes ne peut que réduire l'inertie. Mais le résultat dépend de l'initialisation → c'est un minimum local. D'où n_init=10 pour tester plusieurs départs." },

  # ── NON-SUPERVISÉ : PCA (8 questions) ──

  { id: 116, category: "Non-supervisé", difficulty: "facile",
    question: "Que fait la PCA (Principal Component Analysis) ?",
    choices: ["Réduit le nombre de dimensions en gardant le maximum de variance", "Classifie les données, dans un contexte ML classique", "Regroupe en clusters, dans la plupart des cas", "Supprime les outliers, c'est recommandé par défaut"],
    answer: 0,
    explanation: "PCA projette les données de N dimensions vers P dimensions (P < N) en trouvant les axes de variance maximale. 50 features → 2 ou 3 composantes principales pour visualiser ou accélérer." },

  { id: 117, category: "Non-supervisé", difficulty: "moyen",
    question: "Après PCA, la composante PC1 explique 65% de la variance et PC2 explique 20%. Que garder ?",
    choices: ["Aucune, 85% est trop peu, cette méthode est utilisée dans les pipelines classiques", "Toutes les composantes, c'est un pattern fréquent en deep learning et ML classique", "PC1 + PC2 = 85% de la variance expliquée", "Uniquement PC1, c'est la norme en data science"],
    answer: 2,
    explanation: "85% de variance expliquée avec seulement 2 composantes (au lieu de 50 features) est un excellent compromis. En général, on cherche ≥ 80-95% de variance cumulée." },

  { id: 118, category: "Non-supervisé", difficulty: "moyen",
    question: "Pourquoi faut-il OBLIGATOIREMENT scaler avant la PCA ?",
    choices: ["C'est optionnel, c'est la méthode standard", "PCA maximise la variance", "Pour accélérer, selon les bonnes pratiques", "Pour réduire les NaN, c'est recommandé par défaut"],
    answer: 1,
    explanation: "PCA cherche les directions de variance maximale. Si 'salaire' (en milliers) a plus de variance que 'âge' (en dizaines), PC1 sera 'salaire' déguisé. Le scaling égalise l'importance." },

  { id: 119, category: "Non-supervisé", difficulty: "difficile",
    question: "Les composantes principales sont-elles interprétables ?",
    choices: ["Oui, chaque composante correspond à une feature, dans un contexte de production classique", "Pas directement — chaque composante est un mélange pondéré de TOUTES les features originales", "Oui, grâce aux feature importances, en règle générale, c'est la configuration par défaut de la plupart des frameworks", "Non, elles sont toujours aléatoires, c'est la norme en data science"],
    answer: 1,
    explanation: "PC1 = 0.4×surface + 0.3×rooms + 0.2×age + ... Ce sont des combinaisons linéaires. On perd l'interprétabilité directe. On peut regarder les loadings (pca.components_) pour comprendre ce que chaque PC capture." },

  { id: 120, category: "Non-supervisé", difficulty: "facile",
    question: "PCA est souvent utilisée avant un modèle ML. Pourquoi ?",
    choices: ["Elle remplace le modèle, en règle générale", "Elle améliore toujours les résultats", "Elle réduit les dimensions", "Elle supprime les outliers, comme en machine learning classique"],
    answer: 2,
    explanation: "Moins de features = moins de bruit, moins d'overfitting, entraînement plus rapide. Utile quand il y a beaucoup de features corrélées. Mais attention : on perd en interprétabilité." },

  { id: 121, category: "Non-supervisé", difficulty: "moyen",
    question: "Que montre un <strong>scree plot</strong> (explained variance ratio par composante) ?",
    choices: ["Les corrélations, même en production", "Les clusters, dans la majorité des pipelines", "Le % de variance expliquée par chaque composante", "La loss, indépendamment du dataset"],
    answer: 2,
    explanation: "Le scree plot montre la décroissance de variance. Les premières composantes captent beaucoup, puis ça décroît. On cherche le coude où le gain devient marginal. C'est l'Elbow method de la PCA." },

  { id: 122, category: "Non-supervisé", difficulty: "difficile",
    question: "PCA et K-Means peuvent être combinés. Dans quel ordre ?",
    choices: ["PCA puis K-Means — on réduit les dimensions d'abord pour améliorer le clustering", "L'ordre n'importe pas, dans la plupart des cas, comme le recommande la documentation officielle de sklearn", "K-Means puis PCA, pour tous les modèles, bien que ce soit un choix fréquent chez les débutants", "On ne peut pas les combiner, c'est l'approche courante"],
    answer: 0,
    explanation: "En haute dimension, les distances euclidiennes perdent leur sens (malédiction de la dimensionalité). PCA d'abord réduit le bruit et les dimensions, puis K-Means clusterise dans un espace plus propre." },

  { id: 123, category: "Non-supervisé", difficulty: "moyen",
    question: "Que renvoie <code>pca.explained_variance_ratio_</code> ?",
    choices: ["Les composantes principales, dans la majorité des pipelines", "Un array avec le % de variance expliquée par chaque composante", "Les labels des clusters, pour des raisons de performance", "Les erreurs du modèle, même en production"],
    answer: 1,
    explanation: "Exemple : [0.65, 0.20, 0.08, 0.04, 0.03] → PC1 explique 65%, PC2 20%, etc. La somme cumulée (0.65, 0.85, 0.93...) aide à choisir le nombre de composantes à garder." },

  # ── DEEP LEARNING : NN / MLP (10 questions) ──

  { id: 124, category: "Deep Learning", difficulty: "facile",
    question: "Qu'est-ce qu'un <strong>neurone artificiel</strong> ?",
    choices: ["Un cluster de données, pour des raisons de performance, ce qui est souvent mentionné dans la littérature", "Un programme autonome, c'est recommandé par défaut, comme indiqué dans les tutoriels de référence", "Un calcul : somme pondérée des entrées + biais, passée dans une fonction d'activation", "Un type de feature, dans un contexte ML classique, cette pratique est commune dans l'industrie"],
    answer: 2,
    explanation: "Neurone = activation(Σ(wᵢ × xᵢ) + b). Chaque entrée est multipliée par un poids, on ajoute un biais, puis on applique une activation (ReLU, sigmoïde...). C'est la brique de base du deep learning." },

  { id: 125, category: "Deep Learning", difficulty: "facile",
    question: "Que fait la fonction d'activation <strong>ReLU</strong> ?",
    choices: ["Normalise entre 0 et 1, pour des raisons de performance", "max(0, x) — renvoie 0 si négatif, x si positif", "Calcule la probabilité, même en production", "Renvoie toujours 1, c'est recommandé par défaut"],
    answer: 1,
    explanation: "ReLU est simple et efficace : elle 'éteint' les valeurs négatives (→ 0) et laisse passer les positives. C'est la fonction d'activation par défaut dans les couches cachées car elle ne souffre pas du vanishing gradient." },

  { id: 126, category: "Deep Learning", difficulty: "moyen",
    question: "Pour une classification binaire, quelles activation et loss utiliser en sortie ?",
    choices: ["Sigmoïde + binary_crossentropy", "Softmax + categorical_crossentropy", "Tanh + MAE, c'est recommandé par défaut", "ReLU + MSE, dans la majorité des pipelines"],
    answer: 0,
    explanation: "Sigmoïde compresse en [0, 1] → probabilité de la classe positive. Binary crossentropy mesure l'écart entre la probabilité prédite et le label réel (0 ou 1)." },

  { id: 127, category: "Deep Learning", difficulty: "moyen",
    question: "Pour une classification en 5 classes, quelles activation et loss utiliser ?",
    choices: ["Sigmoïde + binary_crossentropy, quelle que soit la taille du dataset", "Softmax (5 neurones) + categorical_crossentropy", "ReLU + MSE, dans la majorité des pipelines", "Pas d'activation + MAE, selon la documentation sklearn"],
    answer: 1,
    explanation: "Softmax normalise les 5 sorties pour qu'elles somment à 1 → probabilités par classe. Categorical crossentropy compare cette distribution avec le one-hot du vrai label." },

  { id: 128, category: "Deep Learning", difficulty: "moyen",
    question: "Qu'est-ce que le <strong>backpropagation</strong> ?",
    choices: ["L'ajout de couches, même en production, cette pratique est commune dans l'industrie", "L'initialisation des poids, comme en machine learning classique, ce qui est souvent mentionné dans la littérature", "Le chargement des données, même en production, cette pratique est commune dans l'industrie", "L'algorithme qui calcule les gradients de la loss par rapport à chaque poids, couche par couche en remontant"],
    answer: 3,
    explanation: "Le forward pass calcule la prédiction. Le backward pass (backprop) calcule le gradient de la loss pour chaque poids via la règle de la chaîne. Puis l'optimiseur (Adam, SGD) met à jour les poids." },

  { id: 129, category: "Deep Learning", difficulty: "facile",
    question: "Que fait <code>model.compile(optimizer='adam', loss='mse')</code> ?",
    choices: ["Sauvegarde le modèle, selon la documentation sklearn", "Configure l'optimiseur et la fonction de perte AVANT l'entraînement", "Évalue le modèle, même en production, indépendamment du type de modèle utilisé", "Entraîne le modèle, pour des raisons de performance"],
    answer: 1,
    explanation: "compile() configure le modèle sans l'entraîner : quel optimiseur (Adam), quelle loss (MSE), quelles métriques (accuracy). C'est un prérequis avant .fit()." },

  { id: 130, category: "Deep Learning", difficulty: "moyen",
    question: "Un modèle a un train loss de 0.02 et un val loss de 0.45 après 100 epochs. Diagnostic ?",
    choices: ["Underfitting, en règle générale, dans un contexte de production classique", "Modèle parfait, c'est la norme en data science", "Les données sont mauvaises, dans la majorité des pipelines", "Overfitting sévère — le modèle mémorise le train sans généraliser"],
    answer: 3,
    explanation: "Grand écart train/val = overfitting classique. Solutions : early stopping, dropout, réduire le nombre de couches/neurones, augmenter les données, ajouter de la régularisation." },

  { id: 131, category: "Deep Learning", difficulty: "moyen",
    question: "Que fait la couche <code>Dropout(0.3)</code> ?",
    choices: ["Éteint aléatoirement 30% des neurones à chaque batch pendant l'entraînement", "Supprime 30% des données, c'est recommandé par défaut", "Supprime 30% des features, indépendamment du dataset", "Réduit le learning rate de 30%, en règle générale"],
    answer: 0,
    explanation: "Dropout force le réseau à ne pas dépendre d'un seul neurone. À chaque batch, 30% des neurones sont mis à 0 aléatoirement. Le réseau apprend des représentations plus robustes. Désactivé automatiquement en prédiction." },

  { id: 132, category: "Deep Learning", difficulty: "difficile",
    question: "Pourquoi l'optimiseur <strong>Adam</strong> est-il préféré à SGD ?",
    choices: ["Il utilise moins de mémoire, c'est recommandé par défaut, ce qui est souvent mentionné dans la littérature", "Il est plus simple, c'est la norme en data science, quelle que soit la taille du dataset", "Il adapte le learning rate individuellement pour chaque poids en utilisant les moments (moyenne et variance des gradients)", "Il converge toujours au minimum global, selon la documentation sklearn, sans tenir compte des spécificités du problème"],
    answer: 2,
    explanation: "Adam combine Momentum (moyenne mobile des gradients → direction) et RMSProp (variance des gradients → échelle). Chaque poids a son propre learning rate adaptatif. Souvent meilleur que SGD pur, surtout en début d'entraînement." },

  { id: 133, category: "Deep Learning", difficulty: "difficile",
    question: "Qu'est-ce que le <strong>vanishing gradient</strong> et pourquoi ReLU le résout ?",
    choices: ["Avec sigmoïde/tanh, les gradients deviennent quasi-nuls dans les couches profondes", "Un type d'overfitting, pour des raisons de performance", "Un problème de données, c'est recommandé par défaut, c'est une idée reçue mais qui ne s'applique pas ici", "Un problème de mémoire, c'est la méthode standard, c'est une approche courante mais pas optimale ici"],
    answer: 0,
    explanation: "Sigmoïde sature à 0 ou 1 → gradient ~0 → les premières couches n'apprennent plus. ReLU : gradient = 1 si x > 0, gradient = 0 si x < 0. Le signal se propage sans s'évanouir (sauf pour les neurones 'morts' à 0)." },

  # ── DEEP LEARNING : CNN (8 questions) ──

  { id: 134, category: "Deep Learning", difficulty: "facile",
    question: "Pourquoi utilise-t-on des CNN plutôt que des réseaux denses pour les images ?",
    choices: ["Une image 100×100 = 10 000 features en dense", "Les CNN sont plus rapides, comme le recommande la documentation officielle de sklearn", "Les CNN sont plus simples, ce qui est souvent mentionné dans la littérature", "Les réseaux denses ne gèrent pas les images"],
    answer: 0,
    explanation: "Un réseau dense connecte chaque pixel à chaque neurone → explosion de paramètres. Un CNN utilise des filtres locaux (3×3) qui partagent leurs poids → beaucoup moins de paramètres et capte les patterns locaux (bords, textures)." },

  { id: 135, category: "Deep Learning", difficulty: "moyen",
    question: "Que fait une couche de <strong>convolution</strong> ?",
    choices: ["Classifie l'image, c'est la norme en data science", "Aplatit l'image en vecteur, c'est l'approche courante", "Réduit la taille de l'image, c'est la méthode standard", "Applique des filtres (kernels) qui détectent des patterns locaux"],
    answer: 3,
    explanation: "Un filtre 3×3 glisse sur l'image et calcule un produit scalaire à chaque position. Chaque filtre détecte un pattern différent. La première couche détecte des bords, les couches profondes des formes complexes." },

  { id: 136, category: "Deep Learning", difficulty: "moyen",
    question: "Que fait la couche <code>MaxPooling2D(2, 2)</code> ?",
    choices: ["Réduit la taille par 2 en gardant la valeur max dans chaque fenêtre 2×2", "Normalise les pixels, selon les bonnes pratiques", "Augmente le nombre de filtres, c'est la norme en data science", "Ajoute des pixels, c'est la méthode standard, d'après les conventions établies en data science"],
    answer: 0,
    explanation: "MaxPool(2,2) découpe en fenêtres 2×2 et garde le max de chaque fenêtre. L'image passe de 32×32 à 16×16. Effet : réduit les paramètres, rend le modèle plus robuste aux petits déplacements." },

  { id: 137, category: "Deep Learning", difficulty: "moyen",
    question: "Un CNN typique a l'architecture : Conv→Pool→Conv→Pool→Flatten→Dense→Output. Pourquoi Flatten ?",
    choices: ["C'est optionnel, dans la plupart des cas", "Pour augmenter les features, c'est la méthode standard", "Les couches denses attendant un vecteur 1D, pas un tensor 3D", "Pour accélérer, comme en machine learning classique"],
    answer: 2,
    explanation: "Après les convolutions/pooling, les données sont un tensor 3D (hauteur × largeur × filtres). La couche Dense finale nécessite un vecteur 1D pour faire la classification. Flatten aplatit le tensor." },

  { id: 138, category: "Deep Learning", difficulty: "facile",
    question: "Pourquoi normaliser les pixels entre 0 et 1 pour un CNN ?",
    choices: ["Par convention esthétique, dans la majorité des pipelines", "C'est obligatoire techniquement, bien que ce soit un choix fréquent chez les débutants", "Des valeurs [0, 255] causent des gradients instables", "Pour réduire la taille du fichier, cette pratique est commune dans l'industrie"],
    answer: 2,
    explanation: "Pixels bruts [0, 255] → valeurs très grandes → gradients instables. Diviser par 255.0 ramène tout entre [0, 1]. L'entraînement converge plus vite et plus stablement." },

  { id: 139, category: "Deep Learning", difficulty: "difficile",
    question: "Qu'est-ce que le <strong>transfer learning</strong> ?",
    choices: ["Copier les hyperparamètres d'un autre projet, quelle que soit la taille du dataset", "Transférer les données entre machines, c'est la norme en data science", "Utiliser un modèle pré-entraîné (ex: VGG16 sur ImageNet) et l'adapter à sa propre tâche", "Entraîner sur plusieurs GPU, selon la documentation sklearn"],
    answer: 2,
    explanation: "Un modèle entraîné sur ImageNet (1.2M images) a déjà appris à détecter bords, textures, formes. On gèle ces couches et on réentraîne seulement les dernières couches dense sur nos données. Résultat : excellent même avec peu de données." },

  { id: 140, category: "Deep Learning", difficulty: "moyen",
    question: "La <strong>data augmentation</strong> pour les images consiste à :",
    choices: ["Appliquer des transformations (rotation, flip, zoom, crop) pour créer des variantes artificielles", "Supprimer les images floues, même en production, quelle que soit la taille du dataset", "Compresser les images, selon la documentation sklearn, cette méthode est utilisée dans les pipelines classiques", "Télécharger plus d'images, dans un contexte ML classique, d'après les conventions établies en data science"],
    answer: 0,
    explanation: "Un chat retourné reste un chat. En générant des variantes (flip horizontal, rotation ±15°, zoom), on multiplie artificiellement le dataset → moins d'overfitting, meilleure généralisation." },

  { id: 141, category: "Deep Learning", difficulty: "difficile",
    question: "Quelle est la différence entre <code>padding='same'</code> et <code>padding='valid'</code> en Conv2D ?",
    choices: ["Aucune différence, c'est la norme en data science", "'same' ajoute des zéros pour garder la même taille en sortie", "'valid' est plus rapide, c'est la méthode standard", "'same' augmente la résolution, dans la plupart des cas"],
    answer: 1,
    explanation: "Un filtre 3×3 sur une image 32×32 avec padding='valid' donne 30×30 (perte de bord). Avec padding='same', on ajoute des zéros autour → sortie 32×32. 'same' est plus courant car il évite la réduction progressive." },

  # ── DEEP LEARNING : RNN / Séquences (6 questions) ──

  { id: 142, category: "Deep Learning", difficulty: "facile",
    question: "Pourquoi utiliser un RNN plutôt qu'un réseau dense pour des séries temporelles ?",
    choices: ["Le dense ne gère pas les nombres", "Les RNN ont moins de paramètres", "Les RNN tiennent compte de l'ORDRE des données", "Les RNN sont plus rapides, c'est une idée reçue mais qui ne s'applique pas ici"],
    answer: 2,
    explanation: "La température de demain dépend de celle d'hier et avant-hier. Un réseau dense ignore cet ordre. Le RNN traite les données séquentiellement et maintient un 'état caché' qui encode l'historique." },

  { id: 143, category: "Deep Learning", difficulty: "moyen",
    question: "Quelle est la différence entre un RNN simple et un <strong>LSTM</strong> ?",
    choices: ["Le LSTM a des 'portes' (forget, input, output) qui contrôlent quoi retenir/oublier", "Le LSTM est plus rapide, dans la majorité des pipelines", "Le RNN simple est plus moderne, c'est la méthode standard", "Aucune, dans la plupart des cas, c'est une approche courante mais pas optimale ici"],
    answer: 0,
    explanation: "Le RNN simple souffre du vanishing gradient sur les longues séquences. Le LSTM (Long Short-Term Memory) a une cellule de mémoire avec 3 portes qui régulent le flux d'information → il peut retenir des dépendances sur des centaines de timesteps." },

  { id: 144, category: "Deep Learning", difficulty: "moyen",
    question: "Pour prédire le prix d'une action à J+1 avec les 30 derniers jours, quelle shape d'entrée ?",
    choices: ["(30, 30) — matrice carrée, comme en machine learning classique", "(30,) — un vecteur de 30 valeurs, indépendamment du dataset", "(1, 30) — une ligne, c'est la méthode standard", "(batch_size, 30, n_features)"],
    answer: 3,
    explanation: "Les RNN/LSTM attendent un tensor 3D : (batch_size, sequence_length, n_features). Ici : séquence de 30 jours, avec pour chaque jour les features (open, high, low, close, volume...)." },

  { id: 145, category: "Deep Learning", difficulty: "difficile",
    question: "Qu'est-ce qu'un <strong>GRU</strong> et quand le préférer au LSTM ?",
    choices: ["Le GRU est un LSTM simplifié avec 2 portes au lieu de 3", "Le GRU est pour les images, dans la majorité des pipelines", "C'est la même chose, pour tous les modèles", "Le GRU est plus ancien, pour tous les modèles"],
    answer: 0,
    explanation: "GRU fusionne la forget gate et l'input gate en une seule 'update gate'. Moins de paramètres → plus rapide à entraîner. Pour des séquences < 100 timesteps, GRU ≈ LSTM. Pour des séquences très longues, LSTM est souvent meilleur." },

  { id: 146, category: "Deep Learning", difficulty: "moyen",
    question: "Pourquoi faut-il créer des <strong>fenêtres glissantes</strong> pour entraîner un RNN sur une série temporelle ?",
    choices: ["Pour supprimer les outliers, dans un contexte ML classique", "Par convention, indépendamment du dataset", "Pour mélanger les données, selon la documentation sklearn", "Le RNN a besoin de paires"],
    answer: 3,
    explanation: "Série : [10, 12, 11, 15, 14, 16]. Fenêtre de 3 → X=[10,12,11] y=15 ; X=[12,11,15] y=14 ; X=[11,15,14] y=16. On crée des paires input/output pour le supervisé." },

  { id: 147, category: "Deep Learning", difficulty: "difficile",
    question: "Que fait <code>return_sequences=True</code> dans une couche LSTM ?",
    choices: ["Renvoie l'état caché à CHAQUE timestep au lieu de seulement le dernier", "Renvoie les poids, en règle générale, c'est la configuration par défaut de la plupart des frameworks", "Active le dropout, dans la plupart des cas, comme le recommande la documentation officielle de sklearn", "Renvoie la prédiction finale, selon les bonnes pratiques"],
    answer: 0,
    explanation: "Par défaut, LSTM ne renvoie que le dernier état. Avec return_sequences=True, on obtient un état par timestep → output shape (batch, timesteps, units). C'est obligatoire avant une autre couche LSTM." },

  # ── PIPELINE SKLEARN (8 questions) ──

  { id: 148, category: "Pipeline", difficulty: "facile",
    question: "Quel est le principal avantage d'un <code>Pipeline</code> sklearn ?",
    choices: ["Il améliore toujours les résultats, dans la majorité des pipelines, indépendamment du type de modèle utilisé", "Il remplace les modèles, selon la documentation sklearn, dans un contexte de production classique", "Il élimine le data leakage automatiquement et rend le code reproductible en une seule ligne fit/predict", "Il est plus rapide, en règle générale, sans tenir compte des spécificités du problème"],
    answer: 2,
    explanation: "Sans Pipeline, il faut manuellement faire scaler.fit(train) → scaler.transform(test). Risque d'erreur → data leakage. Le Pipeline enchaîne tout : pipe.fit(X_train, y_train) fait fit_transform sur train, transform sur test automatiquement." },

  { id: 149, category: "Pipeline", difficulty: "moyen",
    question: "Que fait <code>ColumnTransformer</code> dans un Pipeline ?",
    choices: ["Renomme les colonnes, c'est recommandé par défaut", "Supprime des colonnes, c'est la norme en data science", "Applique des transformations DIFFÉRENTES selon les colonnes", "Ajoute des colonnes, pour tous les modèles"],
    answer: 2,
    explanation: "ColumnTransformer route les colonnes : les numériques passent par imputer+scaler, les catégorielles par imputer+encoder. Le tout en parallèle. Fini le code spaghetti." },

  { id: 150, category: "Pipeline", difficulty: "moyen",
    question: "Que signifie <code>model__n_estimators</code> dans les paramètres de GridSearchCV ?",
    choices: ["model et n_estimators sont indépendants, indépendamment du dataset", "Deux underscores par hasard, selon les bonnes pratiques, d'après les conventions établies en data science", "C'est une erreur de syntaxe, selon les bonnes pratiques, d'après les conventions établies en data science", "Notation Pipeline : 'model' est le nom de l'étape, 'n_estimators' est le paramètre de cette étape"],
    answer: 3,
    explanation: "Dans Pipeline([('preprocessing', ...), ('model', RandomForest())]), le double underscore navigue : model__n_estimators = le paramètre n_estimators de l'étape nommée 'model'." },

  { id: 151, category: "Pipeline", difficulty: "facile",
    question: "Que fait <code>joblib.dump(pipe, 'pipeline.joblib')</code> ?",
    choices: ["Supprime le pipeline, c'est l'approche courante, c'est un pattern fréquent en deep learning et ML classique", "Sauvegarde TOUT le pipeline (preprocessing + modèle) dans un fichier binaire", "Affiche le pipeline, dans la majorité des pipelines", "Entraîne le pipeline, c'est l'approche courante, c'est un pattern fréquent en deep learning et ML classique"],
    answer: 1,
    explanation: "joblib sérialise l'objet Pipeline complet : le scaler (avec ses moyennes/std), l'encoder (avec ses catégories), le modèle (avec ses poids). En production : joblib.load() + pipe.predict()." },

  { id: 152, category: "Pipeline", difficulty: "moyen",
    question: "Peut-on faire un GridSearchCV directement sur un Pipeline ?",
    choices: ["Seulement sur le modèle, selon la documentation sklearn, indépendamment du type de modèle utilisé", "Non, il faut séparer, même en production, dans un contexte de production classique", "Oui — GridSearchCV(pipe, params, cv=5) teste toutes les combinaisons en respectant le pipeline", "Seulement sur le preprocessing, dans la majorité des pipelines"],
    answer: 2,
    explanation: "GridSearchCV respecte le Pipeline : à chaque fold, le preprocessing est fit sur le train et transform sur le val. Pas de data leakage. On peut même tester des paramètres du preprocessing." },

  { id: 153, category: "Pipeline", difficulty: "difficile",
    question: "Pourquoi un Pipeline élimine-t-il le <strong>data leakage</strong> ?",
    choices: ["Le fit() est fait UNIQUEMENT sur X_train", "Il mélange les données, indépendamment du type de modèle utilisé", "Il utilise plus de mémoire", "Il supprime les NaN, selon la documentation sklearn"],
    answer: 0,
    explanation: "Sans Pipeline, erreur classique : scaler.fit_transform(X) sur tout le dataset, PUIS split. Le scaler a 'vu' le test. Avec Pipeline, le fit() interne ne voit que X_train. C'est la garantie structurelle contre le leakage." },

  { id: 154, category: "Pipeline", difficulty: "moyen",
    question: "Tu veux tester 2 scalers et 3 modèles dans un Pipeline. Combien de combinaisons ?",
    choices: ["6 — 2 × 3 = toutes les combinaisons scaler × modèle", "3, selon les bonnes pratiques, sans tenir compte des spécificités du problème", "2, c'est recommandé par défaut, c'est une idée reçue mais qui ne s'applique pas ici", "5, selon les bonnes pratiques, d'après les conventions établies en data science"],
    answer: 0,
    explanation: "GridSearchCV teste toutes les combinaisons : (StandardScaler + RF, StandardScaler + XGBoost, StandardScaler + SVM, MinMaxScaler + RF, MinMaxScaler + XGBoost, MinMaxScaler + SVM) = 6 combinaisons, chacune cross-validée." },

  { id: 155, category: "Pipeline", difficulty: "difficile",
    question: "Que fait <code>handle_unknown='ignore'</code> dans OneHotEncoder au sein d'un Pipeline ?",
    choices: ["Supprime les lignes inconnues, indépendamment du dataset, mais ce n'est pas la meilleure approche pour ce cas", "Si le test contient une catégorie jamais vue dans le train, elle est encodée en vecteur de zéros au lieu de lever une erreur", "Ignore le OneHotEncoding, comme en machine learning classique, sans tenir compte des spécificités du problème", "Supprime les colonnes inconnues, selon les bonnes pratiques, même si certains praticiens le font en exploration"],
    answer: 1,
    explanation: "En production, un nouvel utilisateur peut venir d'une ville jamais vue dans le train. Sans handle_unknown='ignore', le code plante. Avec, la ville est encodée [0, 0, ..., 0] → le modèle fait de son mieux." },

  # ── MLOPS (14 questions) ──

  { id: 156, category: "MLOps", difficulty: "facile",
    question: "Qu'est-ce qu'une <strong>API</strong> dans le contexte MLOps ?",
    choices: ["Un type de modèle, pour tous les modèles, selon les principes fondamentaux du machine learning", "Un serveur qui reçoit des requêtes HTTP et renvoie des prédictions en JSON", "Un type de base de données, indépendamment du dataset", "Un format de données, en règle générale, dans un contexte de production classique"],
    answer: 1,
    explanation: "L'API est l'interface entre le monde extérieur et ton modèle. Tu envoies les features par HTTP, le serveur charge le modèle, fait la prédiction, et renvoie le résultat en JSON." },

  { id: 157, category: "MLOps", difficulty: "facile",
    question: "Que fait FastAPI par rapport à Flask dans le cadre du Wagon ?",
    choices: ["FastAPI est seulement pour le frontend, selon la documentation sklearn", "FastAPI type automatiquement les paramètres, génère la doc Swagger, et est asynchrone par défaut", "Flask est meilleur en production, c'est recommandé par défaut, quelle que soit la taille du dataset", "C'est la même chose, dans un contexte ML classique, c'est une idée reçue mais qui ne s'applique pas ici"],
    answer: 1,
    explanation: "FastAPI : typage (age: int vérifié automatiquement), doc interactive (/docs), validation, performances async. Flask est plus simple mais il faut tout faire manuellement." },

  { id: 158, category: "MLOps", difficulty: "moyen",
    question: "Pourquoi utiliser <code>app.state.model</code> au lieu de charger le modèle dans la fonction predict() ?",
    choices: ["Ça ne change rien, pour tous les modèles, bien que ce soit un choix fréquent chez les débutants", "app.state est obligatoire, dans un contexte ML classique", "Le modèle est chargé UNE SEULE FOIS au démarrage au lieu de à chaque requête", "C'est plus joli, pour des raisons de performance, cette méthode est utilisée dans les pipelines classiques"],
    answer: 2,
    explanation: "joblib.load() peut prendre 2 secondes. Si tu le mets dans predict(), chaque requête attend 2s. Avec app.state, le load se fait au démarrage et l'objet reste en mémoire. Critique pour les modèles Deep Learning." },

  { id: 159, category: "MLOps", difficulty: "moyen",
    question: "Que fait la commande <code>uvicorn api:app --reload</code> ?",
    choices: ["Teste les endpoints, dans la majorité des pipelines, bien que ce soit un choix fréquent chez les débutants", "Déploie sur le cloud, pour tous les modèles, sans tenir compte des spécificités du problème", "Entraîne le modèle, même en production, comme indiqué dans les tutoriels de référence", "Lance le serveur FastAPI avec rechargement automatique à chaque modification du code"],
    answer: 3,
    explanation: "uvicorn = serveur ASGI. api:app → fichier api.py, objet app (FastAPI). --reload relance le serveur à chaque sauvegarde du code. Ne PAS utiliser --reload en production (performance)." },

  { id: 160, category: "MLOps", difficulty: "facile",
    question: "Quel endpoint génère automatiquement la documentation interactive de FastAPI ?",
    choices: ["/admin, indépendamment du dataset", "/test, comme en machine learning classique", "/api, c'est la norme en data science", "/docs — page Swagger générée automatiquement"],
    answer: 3,
    explanation: "FastAPI crée automatiquement /docs (Swagger UI) et /redoc (documentation alternative). On peut y tester chaque endpoint avec le bouton 'Try it out'. Pas besoin de Postman." },

  { id: 161, category: "MLOps", difficulty: "moyen",
    question: "Qu'est-ce que <strong>Docker</strong> fait concrètement ?",
    choices: ["Il empaquette code + dépendances + OS dans un conteneur isolé", "C'est un cloud provider, même en production", "Il optimise le modèle, dans la plupart des cas", "C'est un outil de versioning, c'est la méthode standard"],
    answer: 0,
    explanation: "'Ça marche sur ma machine' → Docker résout ça. Le conteneur embarque Python, les packages, le modèle, le code. Ce qui marche en local marchera exactement pareil sur Cloud Run, sur le PC du collègue, etc." },

  { id: 162, category: "MLOps", difficulty: "moyen",
    question: "Dans le Dockerfile, pourquoi copier requirements.txt AVANT le code ?",
    choices: ["Pour des raisons de sécurité, indépendamment du dataset", "Par convention, c'est la norme en data science", "C'est obligatoire, indépendamment du dataset", "Pour profiter du cache Docker"],
    answer: 3,
    explanation: "Docker construit en couches. Si la couche 'COPY requirements.txt' n'a pas changé depuis le dernier build, Docker réutilise le cache. Le 'pip install' (lent) est skipé. Seul le code (qui change souvent) est recopié." },

  { id: 163, category: "MLOps", difficulty: "facile",
    question: "Pourquoi utiliser <code>python:3.10-slim</code> plutôt que <code>python:3.10</code> dans le Dockerfile ?",
    choices: ["slim est plus récent, dans un contexte ML classique", "slim a plus de packages, cette méthode est utilisée dans les pipelines classiques", "L'image slim fait ~150 MB au lieu de ~900 MB", "Aucune différence, c'est l'approche courante"],
    answer: 2,
    explanation: "L'image 'complète' inclut des outils système inutiles en production. 'slim' contient juste le nécessaire pour faire tourner Python. Moins de taille = build plus rapide, déploiement plus rapide, moins de surface d'attaque." },

  { id: 164, category: "MLOps", difficulty: "moyen",
    question: "Que fait la variable <code>$PORT</code> dans <code>CMD uvicorn api:app --port $PORT</code> ?",
    choices: ["Fixe toujours le port à 8080", "C'est le port de la base de données", "C'est optionnel, comme en machine learning classique", "Cloud Run injecte dynamiquement le port"],
    answer: 3,
    explanation: "Cloud Run attribue un port dynamiquement et le passe via la variable d'environnement PORT. Si tu hardcodes 8080, l'app ne démarre pas. --port $PORT s'adapte automatiquement." },

  { id: 165, category: "MLOps", difficulty: "moyen",
    question: "Que fait <code>gcloud builds submit --tag gcr.io/PROJECT/my-model</code> ?",
    choices: ["Lance l'API en local, dans la majorité des pipelines", "Supprime le projet, c'est l'approche courante, c'est un pattern fréquent en deep learning et ML classique", "Entraîne le modèle sur GCP, dans un contexte ML classique", "Build l'image Docker dans le cloud GCP et la pousse sur le Container Registry"],
    answer: 3,
    explanation: "Pas besoin d'avoir Docker installé en local. GCP build l'image dans le cloud à partir du Dockerfile + code, puis la stocke sur le Container Registry (gcr.io). Prêt pour Cloud Run." },

  { id: 166, category: "MLOps", difficulty: "facile",
    question: "Qu'est-ce que <strong>Cloud Run</strong> ?",
    choices: ["Un service serverless qui lance ton conteneur Docker et scale automatiquement (0 à N instances)", "Un IDE en ligne, même en production, y compris dans les compétitions Kaggle et projets académiques", "Un GPU cloud, c'est l'approche courante, selon les principes fondamentaux du machine learning", "Un service de stockage, pour tous les modèles, c'est un pattern fréquent en deep learning et ML classique"],
    answer: 0,
    explanation: "Cloud Run = 'donne-moi une image Docker, je m'occupe du reste'. 0 requête = 0 instance (0€). 1000 req/sec = auto-scaling. HTTPS automatique. Tu ne gères aucun serveur." },

  { id: 167, category: "MLOps", difficulty: "moyen",
    question: "Pourquoi ajouter <code>--allow-unauthenticated</code> au déploiement Cloud Run ?",
    choices: ["Pour rendre l'API accessible publiquement sans authentification", "Pour accélérer le déploiement, comme en machine learning classique", "C'est obligatoire, selon la documentation sklearn", "Pour réduire les coûts, dans la majorité des pipelines"],
    answer: 0,
    explanation: "Par défaut, Cloud Run exige une authentification IAM. --allow-unauthenticated permet à n'importe qui d'appeler l'API (utile pour un TP ou un MVP). En production, on protège avec des tokens." },

  { id: 168, category: "MLOps", difficulty: "difficile",
    question: "Pourquoi stocker le modèle sur Google Cloud Storage (GCS) plutôt que dans l'image Docker ?",
    choices: ["GCS est plus rapide, pour tous les modèles", "C'est une question de sécurité, bien que ce soit un choix fréquent chez les débutants", "Docker ne supporte pas les gros fichiers", "Un modèle lourd (> 500 MB) gonfle l'image Docker"],
    answer: 3,
    explanation: "Un modèle Deep Learning de 2 Go dans le Docker = image de 2 Go à build et deployer à chaque changement de code. Mieux : stocker le modèle sur GCS et le télécharger au démarrage (app.state)." },

  { id: 169, category: "MLOps", difficulty: "difficile",
    question: "Tu modifies une feature dans le preprocessing. Que faut-il redéployer ?",
    choices: ["Seulement le modèle, c'est l'approche courante, c'est un pattern fréquent en deep learning et ML classique", "Rien, c'est automatique, selon les bonnes pratiques, sans tenir compte des spécificités du problème", "Seulement l'API, dans la plupart des cas, comme le recommande la documentation officielle de sklearn", "TOUT : réentraîner le modèle avec la nouvelle feature, re-sauvegarder le pipeline, rebuild l'image Docker, redéployer"],
    answer: 3,
    explanation: "Le pipeline contient le preprocessing + le modèle ensemble. Si tu changes le preprocessing (nouvelle feature, nouveau scaler), il faut réentraîner, re-sérialiser (joblib.dump), rebuild Docker, redéployer. C'est la chaîne complète." },

  # ── MÉTHODES ML / TRANSVERSAL (15 questions) ──

  { id: 170, category: "Méthodes ML", difficulty: "facile",
    question: "Quelle est la différence entre <strong>classification</strong> et <strong>régression</strong> ?",
    choices: ["La classification n'utilise que des arbres", "Classification prédit une catégorie", "La régression est plus rapide", "Aucune, c'est la méthode standard"],
    answer: 1,
    explanation: "Classification : la target est discrète (chat/chien, 0/1). Régression : la target est continue (prix, température). Le choix du problème détermine le modèle et les métriques." },

  { id: 171, category: "Méthodes ML", difficulty: "moyen",
    question: "Qu'est-ce que la <strong>malédiction de la dimensionalité</strong> ?",
    choices: ["Des données trop petites, c'est l'approche courante", "En haute dimension, les distances deviennent peu informatives", "Trop de catégories, indépendamment du dataset", "Trop de lignes, selon les bonnes pratiques"],
    answer: 1,
    explanation: "Avec 1000 features, l'espace est si vaste que tous les points sont quasi-équidistants. KNN, K-Means et toute méthode basée sur les distances souffrent. Solutions : PCA, feature selection, ou modèles basés sur les arbres." },

  { id: 172, category: "Méthodes ML", difficulty: "facile",
    question: "Qu'est-ce que l'<strong>overfitting</strong> ?",
    choices: ["Le modèle est trop simple, c'est la méthode standard, c'est une approche courante mais pas optimale ici", "Le modèle mémorise le bruit du train set et ne généralise pas sur de nouvelles données", "Les données sont trop propres, dans la plupart des cas, c'est la pratique standard en machine learning supervisé", "Le modèle met trop de temps, en règle générale, c'est la configuration par défaut de la plupart des frameworks"],
    answer: 1,
    explanation: "Overfitting = bon sur le train, mauvais sur le test. Le modèle a appris les particularités du train (bruit, outliers) au lieu des patterns généraux. Solutions : régularisation, cross-validation, early stopping, plus de données." },

  { id: 173, category: "Méthodes ML", difficulty: "moyen",
    question: "Qu'est-ce que le <strong>compromis biais-variance</strong> ?",
    choices: ["Choisir entre vitesse et précision", "Un modèle simple a un biais fort", "Ça ne concerne que le deep learning", "C'est la même chose que le data leakage"],
    answer: 1,
    explanation: "Biais fort = le modèle est trop simple, il rate le pattern (ex: droite pour une courbe). Variance forte = trop complexe, instable d'un dataset à l'autre. L'objectif est de minimiser l'erreur totale = biais² + variance." },

  { id: 174, category: "Méthodes ML", difficulty: "moyen",
    question: "Tu as 500 lignes et 200 features. Quel risque principal ?",
    choices: ["Calcul trop long, c'est l'approche courante", "Données trop propres, pour des raisons de performance", "Underfitting, dans la plupart des cas", "Overfitting — trop de features pour si peu de données"],
    answer: 3,
    explanation: "Plus de features que de lignes → le modèle peut trouver des patterns accidentels dans le bruit. Solutions : feature selection, PCA pour réduire à ~20 composantes, régularisation forte (Lasso, Ridge)." },

  { id: 175, category: "Méthodes ML", difficulty: "facile",
    question: "Le <code>random_state=42</code> sert à quoi ?",
    choices: ["Fixer la graine aléatoire pour obtenir des résultats reproductibles", "Limiter à 42 itérations, dans un contexte ML classique", "Utiliser 42 features, dans la plupart des cas", "Améliorer les résultats, c'est l'approche courante"],
    answer: 0,
    explanation: "Beaucoup d'algorithmes utilisent l'aléatoire (split, bootstrap, initialisation). random_state fixe le générateur → même code = même résultat. 42 par convention (Hitchhiker's Guide), n'importe quel entier fonctionne." },

  { id: 176, category: "Méthodes ML", difficulty: "moyen",
    question: "Qu'est-ce que le <strong>feature engineering</strong> ?",
    choices: ["Supprimer les features inutiles, indépendamment du dataset, c'est la pratique standard en machine learning supervisé", "Renommer les colonnes, pour des raisons de performance, y compris dans les compétitions Kaggle et projets académiques", "Normaliser les données, dans la plupart des cas, c'est une idée reçue mais qui ne s'applique pas ici", "Créer de nouvelles features à partir des existantes pour donner plus d'information au modèle"],
    answer: 3,
    explanation: "Exemples : surface_par_pièce = surface/rooms, est_weekend = jour in [samedi, dimanche], ancienneté = 2025 − année_construction. Le feature engineering est souvent ce qui fait la plus grande différence de performance." },

  { id: 177, category: "Méthodes ML", difficulty: "difficile",
    question: "Pourquoi la <strong>target encoding</strong> peut-elle provoquer du data leakage ?",
    choices: ["Elle utilise la target (y) pour encoder X", "Elle est trop lente, pour des raisons de performance", "C'est faux, aucun risque, d'après les conventions établies en data science", "Elle crée trop de colonnes, bien que ce soit un choix fréquent chez les débutants"],
    answer: 0,
    explanation: "Target encoding remplace chaque catégorie par la moyenne de y pour cette catégorie. Si calculé sur tout le dataset, le modèle 'voit' y dans X. Solution : calculer uniquement sur le train, ou utiliser le smoothing et le fold encoding." },

  { id: 178, category: "Méthodes ML", difficulty: "moyen",
    question: "Quand utiliser l'<strong>AUC-ROC</strong> comme métrique ?",
    choices: ["Toujours en régression, pour des raisons de performance", "En classification binaire", "Seulement pour le deep learning, indépendamment du dataset", "Quand il y a plus de 3 classes, dans un contexte ML classique"],
    answer: 1,
    explanation: "L'AUC-ROC trace le True Positive Rate vs False Positive Rate pour tous les seuils. AUC = 0.5 → modèle aléatoire. AUC = 1 → parfait. C'est utile quand le seuil optimal n'est pas encore choisi." },

  { id: 179, category: "Méthodes ML", difficulty: "facile",
    question: "Que signifie <code>n_jobs=-1</code> dans sklearn ?",
    choices: ["Aucun job en parallèle, même en production, comme le recommande la documentation officielle de sklearn", "Limite à 1 cœur, c'est la norme en data science", "Désactive le calcul, dans la plupart des cas", "Utilise TOUS les cœurs CPU disponibles pour paralléliser le calcul"],
    answer: 3,
    explanation: "n_jobs=-1 dans GridSearchCV, cross_val_score, ou RandomForest utilise tous les cœurs du processeur. Si tu as 8 cœurs, 8 folds de cross-validation tournent en même temps. Speedup quasi-linéaire." },

  { id: 180, category: "Méthodes ML", difficulty: "moyen",
    question: "Quelle est la différence entre <code>GridSearchCV</code> et <code>RandomizedSearchCV</code> ?",
    choices: ["Grid est pour la régression, même en production", "Aucune, selon les bonnes pratiques", "Grid teste TOUTES les combinaisons", "Randomized est plus précis, c'est la méthode standard"],
    answer: 2,
    explanation: "3 paramètres × 10 valeurs = 1000 combinaisons × 5 folds = 5000 fits. GridSearch les fait tous. RandomizedSearchCV(n_iter=50) n'en teste que 50 → 100× plus rapide, souvent presque aussi bon." },

  { id: 181, category: "Méthodes ML", difficulty: "difficile",
    question: "Qu'est-ce que la <strong>stratification</strong> dans train_test_split ?",
    choices: ["Supprimer les doublons, c'est l'approche courante, c'est un pattern fréquent en deep learning et ML classique", "Garantir que chaque split a les mêmes proportions de classes que le dataset original", "Normaliser les données, en règle générale, dans un contexte de production classique", "Trier les données, c'est la méthode standard, c'est une approche courante mais pas optimale ici"],
    answer: 1,
    explanation: "Si tu as 5% de classe 1, un split aléatoire pourrait mettre 0% de classe 1 dans le test. stratify=y force les mêmes proportions (5%) dans train et test. Critique pour les classes déséquilibrées." },

  { id: 182, category: "Méthodes ML", difficulty: "moyen",
    question: "Accuracy = 97% sur un dataset avec 97% de classe 0. Ce modèle est-il bon ?",
    choices: ["Ça dépend du modèle, pour tous les modèles", "Oui, 97% c'est excellent, pour des raisons de performance", "Non — un modèle qui prédit toujours 0 a aussi 97%", "Il faut plus de données, c'est la norme en data science"],
    answer: 2,
    explanation: "L'accuracy paradox : sur des classes déséquilibrées, l'accuracy est trompeuse. Le modèle ignore peut-être complètement la classe rare. Toujours vérifier le classification_report et la confusion matrix." },

  { id: 183, category: "Méthodes ML", difficulty: "difficile",
    question: "Quel est l'effet de l'<strong>early stopping</strong> en deep learning ?",
    choices: ["Arrêter quand le train_loss atteint 0, y compris dans les compétitions Kaggle et projets académiques", "Arrêter l'entraînement à une heure fixe", "Stopper quand la val_loss ne s'améliore plus depuis N epochs", "Limiter le nombre de couches, c'est l'approche courante"],
    answer: 2,
    explanation: "On surveille la val_loss. Si elle remonte pendant 'patience' epochs (ex: 10), on arrête et on restaure les meilleurs poids. C'est la régularisation la plus simple et efficace en deep learning." },

  { id: 184, category: "Méthodes ML", difficulty: "moyen",
    question: "Pourquoi ne PAS mettre les secrets (clés API, mots de passe) dans le code ?",
    choices: ["Si le code est sur Git, les secrets sont exposés publiquement", "C'est plus lent, pour des raisons de performance", "Les secrets ne marchent que en local, c'est la méthode standard", "C'est une convention sans importance, dans un contexte ML classique"],
    answer: 0,
    explanation: "Un .env est ignoré par Git (.gitignore). Les variables d'environnement sont injectées au runtime. En production, Cloud Run/Heroku gèrent les secrets séparément. Jamais de mot de passe en dur dans le code." },

  # ── QUESTIONS TRANSVERSALES / SCÉNARIOS (16 questions) ──

  { id: 185, category: "Méthodes ML", difficulty: "moyen",
    question: "Tu veux prédire si un client va se désabonner (churn). Quel type de problème ?",
    choices: ["Clustering, dans la majorité des pipelines", "Réduction de dimension", "Régression, dans un contexte ML classique", "Classification binaire"],
    answer: 3,
    explanation: "Churn = oui ou non → 2 classes → classification binaire. Modèles adaptés : Logistic Regression, Random Forest, XGBoost. Métriques : recall (ne pas rater de churners), F1, AUC-ROC." },

  { id: 186, category: "Méthodes ML", difficulty: "moyen",
    question: "Tu as un dataset de 50 lignes et 3 features. Quel modèle choisir ?",
    choices: ["Un modèle simple : régression linéaire/logistique ou KNN", "XGBoost avec 1000 arbres, pour tous les modèles", "Deep Learning avec 10 couches, bien que ce soit un choix fréquent chez les débutants", "CNN, c'est l'approche courante, c'est un pattern fréquent en deep learning et ML classique"],
    answer: 0,
    explanation: "50 lignes = très peu de données. Un modèle complexe va overfitter immédiatement. La régression linéaire/logistique a peu de paramètres et généralise bien même avec peu de données." },

  { id: 187, category: "Méthodes ML", difficulty: "difficile",
    question: "Qu'est-ce qu'un <strong>modèle de base</strong> (baseline) et pourquoi en faut-il un ?",
    choices: ["Un modèle naïf (prédire la moyenne, la classe majoritaire) qui sert de référence minimum à battre", "Un modèle sans features, indépendamment du dataset, bien que ce soit un choix fréquent chez les débutants", "Le premier modèle qu'on essaie, pour tous les modèles, indépendamment du type de modèle utilisé", "Le modèle final en production, pour des raisons de performance, d'après les conventions établies en data science"],
    answer: 0,
    explanation: "Avant tout ML, on établit une baseline : en régression → prédire la moyenne ; en classification → prédire la classe majoritaire. Si ton modèle ne bat pas ça, il n'a rien appris d'utile." },

  { id: 188, category: "Preprocessing", difficulty: "moyen",
    question: "Tu vois que 40% de la colonne 'revenue' est manquante. Que faire ?",
    choices: ["Remplir par 0, c'est la norme en data science, y compris dans les compétitions Kaggle et projets académiques", "Supprimer 40% des lignes, c'est l'approche courante, comme indiqué dans les tutoriels de référence", "Investiguer POURQUOI les données manquent, puis imputer (median) ou créer une feature 'revenue_is_missing'", "Supprimer la colonne sans réfléchir, dans un contexte ML classique, c'est une idée reçue mais qui ne s'applique pas ici"],
    answer: 2,
    explanation: "40% de NaN c'est beaucoup. D'abord comprendre : est-ce aléatoire ou structurel (les non-déclarants) ? Si le fait d'être manquant est informatif, ajouter une colonne binaire 'is_missing'. Puis imputer la médiane pour les valeurs manquantes." },

  { id: 189, category: "Méthodes ML", difficulty: "facile",
    question: "Que fait <code>model.score(X_test, y_test)</code> ?",
    choices: ["Sauvegarde le modèle, pour des raisons de performance", "Affiche les paramètres, indépendamment du dataset", "Renvoie la métrique par défaut", "Entraîne le modèle, pour tous les modèles"],
    answer: 2,
    explanation: "Raccourci sklearn. En régression, score() renvoie le R². En classification, l'accuracy. Pour d'autres métriques (F1, RMSE), utiliser les fonctions de sklearn.metrics." },

  { id: 190, category: "Boosting", difficulty: "moyen",
    question: "Quelle est la fonction de <code>eval_set</code> dans xgb.fit() ?",
    choices: ["Les hyperparamètres, selon la documentation sklearn, indépendamment du type de modèle utilisé", "Le jeu de test final, c'est l'approche courante, y compris dans les compétitions Kaggle et projets académiques", "Un jeu de validation pour surveiller la performance et activer l'early stopping", "Le jeu d'entraînement, dans la majorité des pipelines"],
    answer: 2,
    explanation: "eval_set=[(X_val, y_val)] permet à XGBoost de calculer la loss sur la validation à chaque itération. Combiné à early_stopping_rounds=10, l'entraînement s'arrête quand la val_loss stagne." },

  { id: 191, category: "Deep Learning", difficulty: "moyen",
    question: "Qu'est-ce qu'une <strong>epoch</strong> ?",
    choices: ["Un batch de données, dans la majorité des pipelines", "Une prédiction, c'est la norme en data science", "Un pas de gradient, même en production", "Un passage complet sur TOUT le dataset d'entraînement"],
    answer: 3,
    explanation: "1 epoch = le modèle a vu chaque exemple une fois. Avec 1000 exemples et batch_size=100, 1 epoch = 10 batches = 10 mises à jour des poids. On entraîne typiquement 10 à 200 epochs." },

  { id: 192, category: "Deep Learning", difficulty: "moyen",
    question: "Qu'est-ce que le <strong>batch_size</strong> et comment l'ajuster ?",
    choices: ["Le nombre total d'exemples, c'est la méthode standard", "Le nombre d'epochs, même en production, même si certains praticiens le font en exploration", "Le nombre d'exemples traités avant chaque mise à jour des poids", "Le nombre de couches, pour tous les modèles"],
    answer: 2,
    explanation: "Batch=32 → les poids sont mis à jour après 32 exemples. Petit batch (16, 32) = gradients bruités mais meilleure généralisation. Grand batch (256, 512) = gradients stables mais peut rester bloqué dans des minima." },

  { id: 193, category: "Non-supervisé", difficulty: "moyen",
    question: "Tu appliques K-Means et obtiens des clusters de tailles 450, 380, 15, et 5. Que suspecter ?",
    choices: ["Résultat parfait, en règle générale, c'est la configuration par défaut de la plupart des frameworks", "Il faut augmenter K, même en production, bien que ce soit un choix fréquent chez les débutants", "Les clusters 15 et 5 sont probablement des outliers ou un K trop grand", "C'est toujours normal, pour tous les modèles, comme le recommande la documentation officielle de sklearn"],
    answer: 2,
    explanation: "Des clusters minuscules sont souvent des outliers que K-Means isole. Solutions : essayer un K plus petit, supprimer les outliers d'abord, ou utiliser DBSCAN qui les détecte nativement." },

  { id: 194, category: "Pipeline", difficulty: "moyen",
    question: "Pourquoi le Pipeline est-il important en <strong>production</strong> et pas seulement en exploration ?",
    choices: ["En production, le même preprocessing doit être appliqué à la donnée brute de l'utilisateur", "Il est joli dans le code, c'est la norme en data science, c'est une idée reçue mais qui ne s'applique pas ici", "Il est plus rapide, c'est recommandé par défaut, cette pratique est commune dans l'industrie", "C'est optionnel en production, comme en machine learning classique"],
    answer: 0,
    explanation: "En production, l'utilisateur envoie des données brutes. Le Pipeline fait automatiquement : imputer → scaler (avec les stats du train) → encoder → prédire. Sans Pipeline, il faut tout refaire manuellement → source de bugs." },

  { id: 195, category: "MLOps", difficulty: "moyen",
    question: "Tu changes le code de l'API mais pas le modèle. Que redéployer ?",
    choices: ["Rien, selon les bonnes pratiques", "Seulement le modèle, en règle générale", "Tout depuis l'entraînement, sans tenir compte des spécificités du problème", "Seulement rebuild l'image Docker et redéployer"],
    answer: 3,
    explanation: "Si le modèle (pipeline.joblib) n'a pas changé, on ne réentraîne pas. On rebuild l'image Docker (qui inclut le nouveau code) et on redéploie. Grâce au cache Docker, le pip install est skipé → build rapide." },

  { id: 196, category: "Régression Logistique", difficulty: "difficile",
    question: "La courbe ROC de ton modèle passe par le point (FPR=0.1, TPR=0.9). Que signifie-t-il ?",
    choices: ["FPR=0.1 signifie 10% d'accuracy, comme en machine learning classique", "Au seuil correspondant, le modèle détecte 90% des positifs en acceptant 10% de faux positifs", "Le modèle est mauvais, selon les bonnes pratiques, mais ce n'est pas la meilleure approche pour ce cas", "Le modèle est aléatoire, pour tous les modèles, d'après les conventions établies en data science"],
    answer: 1,
    explanation: "TPR=0.9 → on détecte 90% des vrais positifs (recall). FPR=0.1 → on classe à tort 10% des négatifs comme positifs. C'est un excellent point. La diagonale (TPR=FPR) serait un modèle aléatoire." },

  { id: 197, category: "Arbres & Random Forest", difficulty: "moyen",
    question: "Un Random Forest n'a PAS besoin de scaling. Pourquoi ?",
    choices: ["C'est un bug connu, dans la plupart des cas, c'est la pratique standard en machine learning supervisé", "Les arbres font des splits (if x > seuil), pas des calculs de distance", "Le scaling est intégré, dans la plupart des cas", "C'est faux, il faut toujours scaler, comme le recommande la documentation officielle de sklearn"],
    answer: 1,
    explanation: "Un arbre demande : 'surface > 80 ?' La réponse ne change pas si surface est en m² ou en km². Les arbres (et donc RF, XGBoost) sont invariants au scaling. C'est un de leurs avantages." },

  { id: 198, category: "Deep Learning", difficulty: "difficile",
    question: "Qu'est-ce qu'un <strong>learning rate scheduler</strong> ?",
    choices: ["Un planning d'entraînement, pour tous les modèles, selon les principes fondamentaux du machine learning", "Un mécanisme qui diminue le learning rate au fil des epochs pour affiner la convergence", "Un type de couche, même en production, quelle que soit la taille du dataset", "Un remplacement de l'optimiseur, même en production, quelle que soit la taille du dataset"],
    answer: 1,
    explanation: "Au début, un LR élevé permet d'avancer vite. Plus on s'approche du minimum, plus il faut des petits pas. Le scheduler réduit le LR automatiquement (ex: diviser par 10 tous les 30 epochs) pour converger plus finement." },

  { id: 199, category: "Méthodes ML", difficulty: "moyen",
    question: "Qu'est-ce que le <strong>No Free Lunch theorem</strong> signifie en pratique ?",
    choices: ["Les données sont toujours insuffisantes", "Le deep learning est toujours meilleur", "Il faut payer pour les données, quelle que soit la taille du dataset", "Aucun algorithme n'est universellement le meilleur"],
    answer: 3,
    explanation: "Le théorème dit qu'aucun modèle ne domine tous les autres sur tous les problèmes. XGBoost est souvent bon sur du tabulaire, les CNN sur les images, mais il n'y a pas de certitude. D'où l'importance du benchmarking." },

  { id: 200, category: "Méthodes ML", difficulty: "difficile",
    question: "Tu déploies un modèle de prédiction de prix immobilier. 6 mois plus tard, les prédictions se dégradent. Pourquoi ?",
    choices: ["Le serveur est trop lent, indépendamment du dataset", "Le <strong>data drift</strong>", "Le dataset était trop petit, pour tous les modèles", "Le modèle a été supprimé, c'est la norme en data science"],
    answer: 1,
    explanation: "Data drift = la distribution des données change au fil du temps. Les prix montent, de nouveaux quartiers émergent, les taux d'intérêt changent. Le modèle entraîné sur des données anciennes n'est plus adapté → réentraîner périodiquement." },

)
