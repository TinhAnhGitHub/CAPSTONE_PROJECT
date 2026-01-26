find shared service_asr service_autoshot service_image_embedding service_llm service_text_embedding \
  -type f \
  -not -path "*/beit3/*" \
  -not -path "app/services/unilm/*" \
  -not -path "*/__pycache__/*" \
  -not -name "*.pyc" \
  -not -name "*.pyo" \
  -not -name "*.toml" \
  -not -name "*.lock" \
  -not -name "*.log" \
| while read f; do
    echo "===== $f ====="
    cat "$f"
    echo -e "\n"
done > output.txt