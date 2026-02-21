class HomeController < ApplicationController
  def index
    @current = { slug: "home" }
    @workflows = WORKFLOWS
    @glossary_count = defined?(GLOSSARY) ? GLOSSARY.length : 0
    @methods_count = defined?(ML_METHODS) ? ML_METHODS.length : 0
    @quiz_count = defined?(QUIZ_QUESTIONS) ? QUIZ_QUESTIONS.length : 0

    @parcours = [
      { label: "Préparation", icon: "🧹", color: "green",
        description: "Nettoyer les données brutes (valeurs manquantes, outliers, encodage), puis explorer visuellement les distributions, corrélations et patterns cachés avant de modéliser.",
        workflows: @workflows.select { |w| %w[preprocessing eda].include?(w[:slug]) } },
      { label: "Supervisé", icon: "🎯", color: "blue",
        description: "Apprendre à prédire une valeur continue (prix, salaire) ou une catégorie (spam/pas spam, fraude/légitime) à partir de données étiquetées. Du plus simple au plus puissant.",
        workflows: @workflows.select { |w| %w[linreg logreg trees boosting knn svm].include?(w[:slug]) } },
      { label: "Non-supervisé", icon: "🧩", color: "purple",
        description: "Quand on n'a pas de labels : regrouper des clients similaires (clustering), compresser 50 features en 5 composantes principales (PCA), visualiser en 2D.",
        workflows: @workflows.select { |w| %w[kmeans pca].include?(w[:slug]) } },
      { label: "Deep Learning", icon: "🧠", color: "red",
        description: "Construire des réseaux de neurones couche par couche (Dense), classifier des images avec des convolutions (CNN), traiter du texte et des séries temporelles avec des LSTM (RNN).",
        workflows: @workflows.select { |w| %w[nn cnn rnn].include?(w[:slug]) } },
      { label: "Production", icon: "🚀", color: "yellow",
        description: "Assembler tout le workflow dans un Pipeline sklearn reproductible, puis déployer le modèle en production avec MLflow, Docker et les bonnes pratiques MLOps.",
        workflows: @workflows.select { |w| %w[mlops pipeline].include?(w[:slug]) } },
    ]
  end
end
