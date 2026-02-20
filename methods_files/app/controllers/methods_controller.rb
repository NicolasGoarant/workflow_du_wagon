class MethodsController < ApplicationController
  def index
    @methods = ML_METHODS
    @phases = ML_METHODS.map { |m| m[:phase] }.uniq
    @current = { slug: "methodes" }
  end
end
