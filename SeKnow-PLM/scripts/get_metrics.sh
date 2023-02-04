export DATASETS_PATH="../../datasets_allowed_domains"
python evaluate_multiwoz.py --file "results/datasets_allowed_domains/test-predictions.txt" --use_blacklist > "results/datasets_allowed_domains/test-results.txt"

export DATASETS_PATH="../../datasets_protocol1"
python evaluate_multiwoz.py --file "results/datasets_protocol1/test-predictions.txt" --use_blacklist > "results/datasets_protocol1/test-results.txt"
