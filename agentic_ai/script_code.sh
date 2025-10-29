find agent const core tools\
  -type f \
  -not -path "app/services/unilm/*" \
  -not -path "*/__pycache__/*" \
  -not -name "*.pyc" \
  -not -name "*.pyo" \
| while read f; do
    echo "===== $f ====="
    cat "$f"
    echo -e "\n"
done > output.txt
