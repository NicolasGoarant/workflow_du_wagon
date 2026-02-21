class SearchController < ApplicationController
  def index
    @query = params[:q].to_s.strip.downcase
    @results = []

    return if @query.length < 2

    # Search workflows (title, subtitle, steps)
    WORKFLOWS.each do |w|
      score = 0
      score += 10 if w[:title].downcase.include?(@query)
      score += 5 if w[:subtitle].downcase.include?(@query)
      w[:steps].each do |step|
        score += 3 if step[:title].downcase.include?(@query)
        score += 2 if step[:explain]&.downcase&.include?(@query)
        score += 1 if step[:code_block]&.downcase&.include?(@query)
        step[:code_notes]&.each do |note|
          score += 2 if note[:text]&.downcase&.include?(@query)
        end
      end
      if score > 0
        @results << { type: "workflow", icon: w[:icon], title: w[:title], subtitle: w[:subtitle], url: "/workflows/#{w[:slug]}", score: score }
      end
    end

    # Search glossary
    GLOSSARY.each do |g|
      score = 0
      score += 10 if g[:term].downcase.include?(@query)
      score += 3 if g[:definition].downcase.include?(@query)
      score += 1 if g[:code]&.downcase&.include?(@query)
      if score > 0
        @results << { type: "glossaire", icon: "📖", title: g[:term], subtitle: g[:definition].truncate(80), url: "/glossaire##{g[:term].parameterize rescue g[:term].downcase.gsub(/[^a-z0-9]+/, '-')}", score: score }
      end
    end

    # Search methods
    if defined?(ML_METHODS)
      ML_METHODS.each do |m|
        score = 0
        score += 10 if m[:method].downcase.include?(@query)
        score += 5 if m[:short].downcase.include?(@query)
        score += 3 if m[:explain].downcase.include?(@query)
        if score > 0
          @results << { type: "méthode", icon: "⚙️", title: m[:method], subtitle: m[:short].truncate(80), url: "/methodes", score: score }
        end
      end
    end

    @results.sort_by! { |r| -r[:score] }
    @results = @results.first(15)

    respond_to do |format|
      format.html
      format.json { render json: @results }
    end
  end
end
