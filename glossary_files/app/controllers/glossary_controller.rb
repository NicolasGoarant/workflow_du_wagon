class GlossaryController < ApplicationController
  def index
    @terms = GLOSSARY.sort_by { |t| t[:term].downcase }
    @categories = GLOSSARY.map { |t| t[:category] }.uniq.sort
    @workflows = WORKFLOWS
  end
end
