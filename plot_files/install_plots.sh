#!/bin/bash
# Installation des plots Chart.js — Workflow du Wagon
# Exécuter depuis la racine du projet :
#   bash plot_files/install_plots.sh

set -e
echo "📊 Installation des plots..."

# 1. Copier les données
echo "  → Données des plots..."
cp plot_files/config/initializers/plots_data.rb config/initializers/
cp plot_files/config/initializers/plots_mapping.rb config/initializers/

# 2. Copier le partial
echo "  → Partial de rendu..."
mkdir -p app/views/shared
cp plot_files/app/views/shared/_plot.html.erb app/views/shared/

# 3. CSS
echo "  → CSS..."
cat plot_files/plots.css >> app/assets/stylesheets/application.css

# 4. Ajouter Chart.js CDN dans le layout (avant </head>)
if grep -q "chart.js" app/views/layouts/application.html.erb; then
  echo "  → Chart.js déjà présent ✓"
else
  echo "  → Ajout de Chart.js CDN..."
  sed -i '/<link rel="manifest"/i\  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>' app/views/layouts/application.html.erb
fi

# 5. Ajouter le rendu des plots dans le partial _workflow
if grep -q "STEP_PLOTS" app/views/workflows/_workflow.html.erb; then
  echo "  → Plots déjà dans le partial ✓"
else
  echo "  → Injection des plots dans le partial..."
  # Insert plot rendering after code_notes block, before the closing </div> of flow-step
  sed -i '/code-note-text.*html_safe/,/<% end %>/{ /^        <% end %>$/a\
\
        <%# === PLOTS === %>\
        <% plot_ids = STEP_PLOTS[[workflow[:slug], i]] rescue nil %>\
        <% if plot_ids %>\
          <% plot_ids.each do |pid| %>\
            <%= render "shared/plot", plot_id: pid %>\
          <% end %>\
        <% end %>
  }' app/views/workflows/_workflow.html.erb
fi

# 6. Vider le cache
rm -rf tmp/cache

echo ""
echo "✅ Plots installés ! 📊"
echo "16 graphiques interactifs Chart.js ajoutés aux workflows."
echo "Relance : rails server"
