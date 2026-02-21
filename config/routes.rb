Rails.application.routes.draw do
  root "workflows#index"
  get "recherche", to: "search#index", as: :recherche
  get "methodes", to: "methods#index", as: :methodes
  get "glossaire", to: "glossary#index", as: :glossaire
  get "workflows/:slug", to: "workflows#show", as: :workflow
end
