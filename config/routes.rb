Rails.application.routes.draw do
  root "home#index"
  get "comprendre/modele", to: "concepts#model", as: :concept_model
  get "quiz", to: "quiz#index", as: :quiz
  get "recherche", to: "search#index", as: :recherche
  get "methodes", to: "methods#index", as: :methodes
  get "glossaire", to: "glossary#index", as: :glossaire
  get "cheatsheet", to: "tools#cheatsheet", as: :cheatsheet
  get "quel-modele", to: "tools#chooser", as: :chooser
  get "comparateur", to: "tools#comparator", as: :comparator
  get "workflows/:slug", to: "workflows#show", as: :workflow
end
