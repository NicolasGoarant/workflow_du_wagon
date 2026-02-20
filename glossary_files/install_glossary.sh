#!/bin/bash
# ============================================================
# INSTALLATION DU GLOSSAIRE — Workflow du Wagon
# ============================================================
# Exécuter depuis la racine du projet :
#   bash install_glossary.sh
# ============================================================

set -e
echo "📖 Installation du glossaire..."

# 1. Copier les fichiers
echo "  → Données du glossaire..."
cp glossary_files/config/initializers/glossary_data.rb config/initializers/

echo "  → Contrôleur..."
cp glossary_files/app/controllers/glossary_controller.rb app/controllers/

echo "  → Vues..."
cp glossary_files/app/views/glossary/index.html.erb app/views/glossary/
mkdir -p app/views/shared
cp glossary_files/app/views/shared/_glossary_drawer.html.erb app/views/shared/

echo "  → CSS..."
cat glossary_files/glossary.css >> app/assets/stylesheets/application.css

# 2. Ajouter la route
if grep -q "glossary" config/routes.rb; then
  echo "  → Route déjà présente ✓"
else
  echo "  → Ajout de la route..."
  sed -i '/root "workflows#index"/a\  get "glossaire", to: "glossary#index", as: :glossaire' config/routes.rb
fi

# 3. Ajouter le drawer dans le layout (avant </main>)
if grep -q "glossary_drawer" app/views/layouts/application.html.erb; then
  echo "  → Drawer déjà dans le layout ✓"
else
  echo "  → Ajout du drawer dans le layout..."
  sed -i 's|</main>|</main>\n\n  <%= render "shared/glossary_drawer" %>|' app/views/layouts/application.html.erb
fi

# 4. Ajouter le lien Glossaire dans la nav desktop
if grep -q "glossaire" app/views/layouts/application.html.erb; then
  echo "  → Lien nav déjà présent ✓"
else
  echo "  → Ajout du lien dans la nav desktop..."
  sed -i 's|</nav><!-- desktop -->|  <a href="/glossaire" class="nav-btn nav-btn--glossary"><span class="nav-icon">📖</span><span class="nav-label">Glossaire</span></a>\n  </nav><!-- desktop -->|' app/views/layouts/application.html.erb
fi

# 5. Ajouter le lien dans le drawer mobile
if grep -q "glossaire" app/views/layouts/application.html.erb | grep -q "drawer-link"; then
  echo "  → Lien drawer déjà présent ✓"
else
  echo "  → Ajout du lien dans le drawer mobile..."
  sed -i 's|</div><!-- drawer-sections -->|  <div class="drawer-cat">Outils</div>\n        <a href="/glossaire" class="drawer-link"><span class="drawer-icon">📖</span><span>Glossaire</span></a>\n      </div><!-- drawer-sections -->|' app/views/layouts/application.html.erb
fi

# 6. Vider le cache
rm -rf tmp/cache

echo ""
echo "✅ Glossaire installé !"
echo ""
echo "Relance le serveur : rails server"
echo "Accès : http://localhost:3000/glossaire"
echo "Ou clique sur le bouton 📖 flottant depuis n'importe quel workflow"
