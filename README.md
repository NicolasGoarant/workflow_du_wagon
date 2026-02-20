# 🧠 Workflow du Wagon

Plateforme de référence Data Science — 15 workflows Le Wagon avec code détaillé, outputs concrets et annotations ligne par ligne.

## Setup (IMPORTANT : suivre dans l'ordre)

```bash
# 1. Extraire
tar xzf workflow_du_wagon.tar.gz
cd workflow_du_wagon

# 2. SUPPRIMER le cache (crucial si ancienne version)
rm -rf tmp/cache public/assets

# 3. Installer
bundle install

# 4. Créer la base
rails db:create

# 5. Lancer
rails server
```

Ouvrir **http://localhost:3000**

> Si l'ancien design s'affiche : Cmd+Shift+R (Mac) ou navigation privée.
