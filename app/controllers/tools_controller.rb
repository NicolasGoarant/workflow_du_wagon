class ToolsController < ApplicationController
  def cheatsheet
    @current = { slug: "cheatsheet" }
  end

  def chooser
    @current = { slug: "chooser" }
  end

  def comparator
    @models = COMPARATOR_MODELS
    @current = { slug: "comparator" }
  end
end
