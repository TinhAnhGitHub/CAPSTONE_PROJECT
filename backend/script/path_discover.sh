find app/common app/controller app/model app/repository app/schema app/service app/main.py app/api app/core \
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
