class HomeController < ApplicationController
  def index
    @current = { slug: "home" }
    @workflows = WORKFLOWS
    @glossary_count = defined?(GLOSSARY) ? GLOSSARY.length : 0
    @methods_count = defined?(ML_METHODS) ? ML_METHODS.length : 0
    @quiz_count = defined?(QUIZ_QUESTIONS) ? QUIZ_QUESTIONS.length : 0

    @parcours = [
      { label: "Préparation", icon: "🧹", color: "green",
        description: "Nettoyer, explorer, comprendre les données",
        workflows: @workflows.select { |w| %w[preprocessing eda].include?(w[:slug]) } },
      { label: "Supervisé", icon: "🎯", color: "blue",
        description: "Prédire une valeur ou une classe",
        workflows: @workflows.select { |w| %w[linreg logreg trees boosting knn svm].include?(w[:slug]) } },
      { label: "Non-supervisé", icon: "🧩", color: "purple",
        description: "Découvrir des groupes, réduire les dimensions",
        workflows: @workflows.select { |w| %w[kmeans pca].include?(w[:slug]) } },
      { label: "Deep Learning", icon: "🧠", color: "red",
        description: "Réseaux de neurones, images, séquences",
        workflows: @workflows.select { |w| %w[nn cnn rnn].include?(w[:slug]) } },
      { label: "Production", icon: "🚀", color: "yellow",
        description: "Pipeline, MLOps, déploiement",
        workflows: @workflows.select { |w| %w[mlops pipeline].include?(w[:slug]) } },
    ]
  end
end
