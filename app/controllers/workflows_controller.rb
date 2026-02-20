class WorkflowsController < ApplicationController

  def index
    @current = @workflows.first
  end

  def show
    @current = @workflows.find { |w| w[:slug] == params[:slug] }
    redirect_to root_path unless @current
  end

  private

end
