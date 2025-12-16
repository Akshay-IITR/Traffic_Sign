The Folder "Radar_Federated" contains the following sub-folders:-
1) Dataset_Path: This folder contains various json files of the shared dataset, 
   i.e., test_dataset, train_dataset, valid_dataset, feature_norm
2) model_weights_otc: It contains the shared DeepLoc models.
3) model_weights_otc_cfd: It contains the shared ConfidNet models.
4) Fed_Model: Our proposed federated models, both DeepLoc and ConfidNet, are saved in this folder.
5) Output: (a) This has the final confusion matrix calculated post-processing steps on our FL model. 
           (b) Recorded video of our simulation

Description of the Files inside Folder "20251106_Model_Details":-
1) ConfidNet_Model.txt - The file is a sample shared code for defining the confidnet model, 
                         the custom loss function, and its training pipeline. Contains the following functions:-
                         (a) def confidence_estimation_appendix(): The training pipeline for confidnet model
                         (b) class ConfidNet(): The function defining the ConfidNet model architecture
                         (c) def confid_obstacle_loss4(): The customized loss function

2) DeepLoc_Model.txt - The file is a sample shared code for deeploc pipeline and contains the following:-
                       class StackedPointNetMask(): The function defining the DeepLoc model architecture
                       def _training(): The training pipeline for the DeepLoc model

3) Input_Definition.txt - This file contains the Meta Data/Information required for running the respective codes
4) pip_requirements.txt - This has all the required package details

7) federated_pipeline.py - This is the main and complete code for running the Radar Project via Federated Learning pipeline.

****************Key function definitions and their description****************
a) confid_obstacle_loss4(...) — custom ConfidNet loss that ignores OOD frames and trains confidence to predict whether predicted class corresponds to obstacle mask.
b) attach_confidnet_head_to_backbone(...) — create ConfidNet from a DeepLoc backbone.
c) stratified_uniform_split(...) — server-side splitting into client shards (stratified when labels exist).
d) weighted_average_weights(...) — FedAvg aggregation (sample-weighted).
e) federated_pipeline(args) — main orchestration of the full federated training pipeline.

****************Example command-line to run the script*********************

The script filename is federated_pipeline.py, a minimal run command looks like with default values:

python federated_pipeline.py \
  --train_json /path/to/train_dataset.json
  --valid_json /path/to/valid_dataset.json
  --test_json  /path/to/test_dataset.json
  --model_path /path/to/best_otc_model_with_architecture.h5
  --save_dir ./Fed_Model
  --batch_size 10
  --num_clients 5
  --frac_clients 0.6
  --global_rounds 5
  --deeploc_local_epochs 5
  --confid_local_epochs 2
  --deeploc_lr 1e-5
  --confid_final_lr 1e-5

**************************************
Summary of the Script: 
**************************************

This repository contains a server-driven Federated Learning pipeline that trains a DeepLoc classification backbone 
and an attached ConfidNet head to estimate prediction confidence for obstacle detection; the main entrypoint is federated_deeploc_confid.py 
which expects train/valid/test JSON datasets and an initial DeepLoc model and performs sample-weighted FedAvg across clients, saving the 
best global ConfidNet and extracted DeepLoc backbone to --save_dir. Usage example: 

python federated_pipeline.py --train_json /path/train.json --valid_json /path/valid.json --test_json /path/test.json 
--model_path /path/best_otc_model_with_architecture.h5 --save_dir ./Fed_Model --batch_size 10 --num_clients 5 --global_rounds 5

— the loader will infer missing metadata (feature names, object counts) from the first frame but will warn on inconsistencies. 
Outputs: saved model files (best_global_confidnet.h5, best_global_deeploc.h5, per-client best DeepLocs) 
and console logs of per-round/client training and evaluation metrics.

8) utils.py - This is the complete code for inferencing the Radar Project on Federated Model after applying post processing approach.

*************************************************************
Summary of the Script: To run simply call "python utils.py"
*************************************************************

-- Read a dataset stored as a JSON file and convert it into NumPy arrays (load_json_as_numpy_dataset).
-- Load per-feature normalization values from a feature_norm.json.
-- Prepare a tf.data.Dataset (normalize, zero-out all-NaN object rows) and run model.predict() to obtain:
-- pred_vector: per-sample class scores / logits or probabilities.
-- confidence: per-sample scalar confidence.
-- Apply smoothing (IIR lowpass) over time to both the pred_vector and confidence.
-- Apply softmax to convert scores into probabilities.
-- Replace predictions whose confidence is below a threshold with either the last confident prediction or a default one-hot category.
-- Compute confusion matrix and classification report.
-- Save two confusion matrix images into output/:
   *** FL_Confusion_Matrix_Count.png (counts heatmap)
   *** FL_Confusion_Matrix_Accuracy.png (per-row percent / accuracy heatmap)

**************************************
Key files & inputs required
**************************************
-- Dataset_Path/test_dataset.json — dataset JSON
-- Dataset_Path/feature_norm.json — per-feature mean/std JSON used by extract_feature_norm().
-- Fed_Model/best_global_confidnet.h5 — pretrained Keras model file.
-- Custom model classes used at load: Mask, MaskedGlobalMaxPool1D, Repeat
-- Output directory: output/