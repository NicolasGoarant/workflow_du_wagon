class ApplicationController < ActionController::Base
  before_action :load_workflows

  private

  def load_workflows
    @workflows = Workflow.all
  end
end
