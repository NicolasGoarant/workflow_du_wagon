#!/bin/bash
echo "🧹 Nettoyage du cache..."
rm -rf tmp/cache tmp/pids tmp/sockets
rm -rf public/assets

echo "📦 Installation des gems..."
bundle install

echo "🗄️ Création de la base de données..."
rails db:create 2>/dev/null || true

echo ""
echo "✅ Prêt ! Lance le serveur avec :"
echo "   rails server"
echo ""
echo "Puis ouvre http://localhost:3000"
echo ""
echo "💡 Si l'ancien CSS s'affiche encore :"
echo "   - Vide le cache navigateur (Cmd+Shift+R sur Mac)"
echo "   - Ou ouvre un onglet privé"
