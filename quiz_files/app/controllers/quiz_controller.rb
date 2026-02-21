class QuizController < ApplicationController
  def index
    @questions = QUIZ_QUESTIONS
    @categories = QUIZ_QUESTIONS.map { |q| q[:category] }.uniq
    @current = { slug: "quiz" }
  end
end
