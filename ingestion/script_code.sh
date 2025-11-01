find prefect_agent \
  -type f \
  -not -path "*/__pycache__/*" \
  -not -path "*/logs/*" \
  -not -path "*/venv/*" \
  -not -name "*.pyc" \
  -not -name "*.pyo" \
  -not -name "*weight*" \
| while read -r f; do
    echo "===== $f ====="
    cat "$f"
    echo -e "\n"
done > output.txt
