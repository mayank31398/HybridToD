import os

model_list = ["datasets_allowed_domains", "datasets_protocol1", "datasets_fully_unstructured"]
data_list = ["datasets_allowed_domains", "datasets_protocol1", "datasets_fully_unstructured"]

def Execute(cmd: str):
    print(cmd)
    print()
    os.system(cmd)

for model in model_list:
    for data in data_list:
        cmd = "mkdir -p output/" + model + "/" + data + "/joint_model"
        Execute(cmd)

        cmd = "jbsub -q x86_24h -cores 1x4+1 -mem 64G -err output/" + model + "/" + data + "/joint_model/err.log -out output/" + model + "/" + data + "/joint_model/out.log python -W ignore src1/main.py --preprocessed_data_path " + data + "/preprocessed_data --model_path models/" + model + "/joint_model --output_file output/" + model + "/" + data + "/joint_model/test.json --eval_file " + data +"/preprocessed_data/test.json --checkpoint best"
        Execute(cmd)

        cmd = "mkdir -p output/" + model + "/" + data + "/separate_models"
        Execute(cmd)

        cmd = "jbsub -q x86_24h -cores 1x4+1 -mem 64G -err output/" + model + "/" + data + "/separate_models/err.log -out output/" + model + "/" + data + "/separate_models/out.log python -W ignore src1/eval_separate_models.py --preprocessed_data_path " + data + "/preprocessed_data --entity_selection_model_path models/" + model + "/entity_selection --response_generation_model_path models/" + model + "/response_generation --output_file output/" + model + "/" + data + "/separate_models/test.json --eval_file " + data +"/preprocessed_data/test.json --checkpoint best"
        Execute(cmd)

        cmd = "python -W ignore src1/eval.py --output_file output/" + model + "/" + data + "/joint_model/test.json --raw_db_path datasets_allowed_domains/preprocessed_data --score_file output/" + model + "/" + data + "/joint_model/score_test.json --slot_value_file output/" + model + "/" + data + "/joint_model/slot_values.json"
        Execute(cmd)

        cmd = "python -W ignore src1/eval.py --output_file output/" + model + "/" + data + "/joint_model/test.json --raw_db_path datasets_allowed_domains/preprocessed_data --score_file output/" + model + "/" + data + "/joint_model/score_test_unique.json --slot_value_file output/" + model + "/" + data + "/joint_model/slot_values_unique.json --use_unique_slots"
        Execute(cmd)

        cmd = "python -W ignore src1/eval.py --output_file output/" + model + "/" + data + "/joint_model/test.json --raw_db_path datasets_allowed_domains/preprocessed_data --score_file output/" + model + "/" + data + "/joint_model/score_test_unique_no_name.json --slot_value_file output/" + model + "/" + data + "/joint_model/slot_values_unique_no_name.json --use_unique_slots --no_name"
        Execute(cmd)

        cmd = "python -W ignore src1/eval.py --output_file output/" + model + "/" + data + "/separate_models/test.json --raw_db_path datasets_allowed_domains/preprocessed_data --score_file output/" + model + "/" + data + "/separate_models/score_test.json --slot_value_file output/" + model + "/" + data + "/separate_models/slot_values.json"
        Execute(cmd)

        cmd = "python -W ignore src1/eval.py --output_file output/" + model + "/" + data + "/separate_models/test.json --raw_db_path datasets_allowed_domains/preprocessed_data --score_file output/" + model + "/" + data + "/separate_models/score_test_unique.json --slot_value_file output/" + model + "/" + data + "/separate_models/slot_values_unique.json --use_unique_slots"
        Execute(cmd)

        cmd = "python -W ignore src1/eval.py --output_file output/" + model + "/" + data + "/separate_models/test.json --raw_db_path datasets_allowed_domains/preprocessed_data --score_file output/" + model + "/" + data + "/separate_models/score_test_unique_no_name.json --slot_value_file output/" + model + "/" + data + "/separate_models/slot_values_unique_no_name.json --use_unique_slots --no_name"
        Execute(cmd)

for data in data_list:
    cmd = "python -W ignore src1/eval.py --output_file output/" + data + "/" + data + "/SeKnow/test.json --raw_db_path datasets_allowed_domains/preprocessed_data --score_file output/" + data + "/" + data + "/SeKnow/score_test.json --slot_value_file output/" + data + "/" + data + "/SeKnow/slot_values.json"
    Execute(cmd)

    cmd = "python -W ignore src1/eval.py --output_file output/" + data + "/" + data + "/SeKnow/test.json --raw_db_path datasets_allowed_domains/preprocessed_data --score_file output/" + data + "/" + data + "/SeKnow/score_test_unique.json --slot_value_file output/" + data + "/" + data + "/SeKnow/slot_values_unique.json --use_unique_slots"
    Execute(cmd)

    cmd = "python -W ignore src1/eval.py --output_file output/" + data + "/" + data + "/SeKnow/test.json --raw_db_path datasets_allowed_domains/preprocessed_data --score_file output/" + data + "/" + data + "/SeKnow/score_test_unique_no_name.json --slot_value_file output/" + data + "/" + data + "/SeKnow/slot_values_unique_no_name.json --use_unique_slots --no_name"
    Execute(cmd)
