require_relative "boot"
require "rails/all"
Bundler.require(*Rails.groups)

module WorkflowDuWagon
  class Application < Rails::Application
    config.load_defaults 7.1
    config.time_zone = "Europe/Paris"
    config.i18n.default_locale = :fr
  end
end
