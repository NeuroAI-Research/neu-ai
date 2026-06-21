# chmod +x build.sh
set -e
OUT_DIR="/Users/ricky/tesla-cat/NeuroAI-Research.github.io"
rm -rf "$OUT_DIR"/*

mkdocs build -f ./mkdocs.yml -d "$OUT_DIR"

cd "$OUT_DIR"
git add .
git commit -m "update $(date +'%Y-%m-%d %H:%M:%S')" 
git push origin main
