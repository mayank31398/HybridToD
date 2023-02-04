mkdir -p "datasets_fully_unstructured/multiwoz-2.1"
python Scripts/create_fully_unstructured.py --sample_statements "Scripts/sample_statements.json" --kb "datasets_allowed_domains/multiwoz-2.1/document_base.json" --dump_path "datasets_fully_unstructured/multiwoz-2.1"


rm -rf "datasets_allowed_domains/preprocessed_data"
python -W ignore src1/main.py --preprocessed_data_path "datasets_allowed_domains/preprocessed_data" --params_file "configs/no_explanation.json" --preprocess_only --raw_data_path "datasets_allowed_domains/multiwoz-2.1" --new_data_path "datasets_allowed_domains/multiwoz-2.1"

rm -rf "datasets_protocol1/preprocessed_data"
python -W ignore src1/main.py --preprocessed_data_path "datasets_protocol1/preprocessed_data" --params_file "configs/no_explanation.json" --preprocess_only --raw_data_path "datasets_allowed_domains/multiwoz-2.1" --new_data_path "datasets_protocol1/multiwoz-2.1"

rm -rf "datasets_fully_unstructured/preprocessed_data"
python -W ignore src1/main.py --preprocessed_data_path "datasets_fully_unstructured/preprocessed_data" --params_file "configs/no_explanation.json" --preprocess_only --raw_data_path "datasets_allowed_domains/multiwoz-2.1" --new_data_path "datasets_fully_unstructured/multiwoz-2.1"



# train
mkdir -p "models/datasets_allowed_domains/joint_model"
jbsub -name "joint_model_datasets_allowed_domains" -q x86_24h -cores 1x4+1 -mem 64G -err "models/datasets_allowed_domains/joint_model/err.log" -out "models/datasets_allowed_domains/joint_model/out.log" python src1/main.py --params_file "configs/no_explanation.json" --model_path "models/datasets_allowed_domains/joint_model" --preprocessed_data_path "datasets_allowed_domains/preprocessed_data"

# train
mkdir -p "models/datasets_protocol1/joint_model"
jbsub -name "joint_model_datasets_protocol1" -q x86_24h -cores 1x4+1 -mem 64G -err "models/datasets_protocol1/joint_model/err.log" -out "models/datasets_protocol1/joint_model/out.log" python src1/main.py --params_file "configs/no_explanation.json" --model_path "models/datasets_protocol1/joint_model" --preprocessed_data_path "datasets_protocol1/preprocessed_data"

# train
mkdir -p "models/datasets_fully_unstructured/joint_model"
jbsub -name "joint_model_datasets_fully_unstructured" -q x86_24h -cores 1x4+1 -mem 64G -err "models/datasets_fully_unstructured/joint_model/err.log" -out "models/datasets_fully_unstructured/joint_model/out.log" python src1/main.py --params_file "configs/no_explanation.json" --model_path "models/datasets_fully_unstructured/joint_model" --preprocessed_data_path "datasets_fully_unstructured/preprocessed_data"



# train entity_selection
mkdir -p "models/datasets_allowed_domains/entity_selection"
jbsub -name "entity_selection_datasets_allowed_domains" -q x86_24h -cores 1x4+1 -mem 64G -err "models/datasets_allowed_domains/entity_selection/err.log" -out "models/datasets_allowed_domains/entity_selection/out.log" python src1/main.py --params_file "configs/entity_selection.json" --model_path "models/datasets_allowed_domains/entity_selection" --preprocessed_data_path "datasets_allowed_domains/preprocessed_data"

# train entity_selection
mkdir -p "models/datasets_protocol1/entity_selection"
jbsub -name "entity_selection_datasets_protocol1" -q x86_24h -cores 1x4+1 -mem 64G -err "models/datasets_protocol1/entity_selection/err.log" -out "models/datasets_protocol1/entity_selection/out.log" python src1/main.py --params_file "configs/entity_selection.json" --model_path "models/datasets_protocol1/entity_selection" --preprocessed_data_path "datasets_protocol1/preprocessed_data"

# train entity_selection
mkdir -p "models/datasets_fully_unstructured/entity_selection"
jbsub -name "entity_selection_datasets_fully_unstructured" -q x86_24h -cores 1x4+1 -mem 64G -err "models/datasets_fully_unstructured/entity_selection/err.log" -out "models/datasets_fully_unstructured/entity_selection/out.log" python src1/main.py --params_file "configs/entity_selection.json" --model_path "models/datasets_fully_unstructured/entity_selection" --preprocessed_data_path "datasets_fully_unstructured/preprocessed_data"



# train response_generation
mkdir -p "models/datasets_allowed_domains/response_generation"
jbsub -name "response_generation_datasets_allowed_domains" -q x86_24h -cores 1x4+1 -mem 64G -err "models/datasets_allowed_domains/response_generation/err.log" -out "models/datasets_allowed_domains/response_generation/out.log" python src1/main.py --params_file "configs/response_generation.json" --model_path "models/datasets_allowed_domains/response_generation" --preprocessed_data_path "datasets_allowed_domains/preprocessed_data"

# train response_generation
mkdir -p "models/datasets_protocol1/response_generation"
jbsub -name "response_generation_datasets_protocol1" -q x86_24h -cores 1x4+1 -mem 64G -err "models/datasets_protocol1/response_generation/err.log" -out "models/datasets_protocol1/response_generation/out.log" python src1/main.py --params_file "configs/response_generation.json" --model_path "models/datasets_protocol1/response_generation" --preprocessed_data_path "datasets_protocol1/preprocessed_data"

# train response_generation
mkdir -p "models/datasets_fully_unstructured/response_generation"
jbsub -name "response_generation_datasets_fully_unstructured" -q x86_24h -cores 1x4+1 -mem 64G -err "models/datasets_fully_unstructured/response_generation/err.log" -out "models/datasets_fully_unstructured/response_generation/out.log" python src1/main.py --params_file "configs/response_generation.json" --model_path "models/datasets_fully_unstructured/response_generation" --preprocessed_data_path "datasets_fully_unstructured/preprocessed_data"