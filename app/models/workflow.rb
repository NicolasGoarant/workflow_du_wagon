class Workflow
  def self.all
    WORKFLOWS
  end

  def self.find_by_slug(slug)
    all.find { |w| w[:slug] == slug }
  end
end
