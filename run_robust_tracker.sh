cat > render_results_video.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
python3 render_results_video.py "$@"
EOF

chmod +x render_results_video.sh