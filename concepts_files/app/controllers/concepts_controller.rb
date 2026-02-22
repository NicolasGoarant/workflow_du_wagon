class ConceptsController < ApplicationController
  def model
    @concepts = MODEL_CONCEPTS
    @current = { slug: "concepts" }
  end
end
