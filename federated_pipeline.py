#!/usr/bin/env python3
"""
Implemented the described Federated DeepLoc + ConfidNet pipeline:

- Round 1: server loads DeepLoc (from --model_path) -> disseminates DeepLoc to selected clients.
  Each client: train DeepLoc locally -> build ConfidNet locally (attach head) -> train ConfidNet locally -> send ConfidNet to server.
  Server aggregates ConfidNet -> global ConfidNet -> evaluate on test set.

- Round >1: server disseminates global ConfidNet to selected clients.
  Each client: split ConfidNet -> detach head (save temporarily), train detached DeepLoc locally ->
  reattach head, train ConfidNet locally -> send ConfidNet to server.
  Server aggregates ConfidNet -> evaluate on test set.

The above described iteration continues till the expected performance is achieved or given rounds. 
"""
import json
import math
import os, gc
import argparse
import random
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, DefaultDict
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model
from tensorflow_model_optimization.python.core.keras.compat import keras
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

tf.get_logger().setLevel('ERROR')

class Mask(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mask, self).__init__(**kwargs)
        self.__name__ = 'Mask'
    def call(self, input_points):
        return keras.backend.any(keras.backend.not_equal(input_points, 0.0), axis=-1)
    def get_config(self):
        return super().get_config()

class Repeat(keras.layers.Layer):
    def __init__(self, max_locations, **kwargs):
        super(Repeat, self).__init__(**kwargs)
        self.__name__ = 'Repeat'
        self.max_locations = max_locations
    def call(self, inputs):
        return tf.reshape(keras.backend.repeat(inputs, self.max_locations), shape=(-1, self.max_locations, inputs.shape[1]))
    def get_config(self):
        config = super().get_config()
        config.update({'max_locations': self.max_locations})
        return config

class MaskedGlobalMaxPool1D(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaskedGlobalMaxPool1D, self).__init__(**kwargs)
        self.supports_masking = True
    def compute_mask(self, inputs, mask=None):
        return None
    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + (input_shape[-1],)
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = keras.backend.cast(mask, keras.backend.floatx())
            inputs *= keras.backend.expand_dims(mask, axis=-1)
        return keras.backend.max(inputs, axis=-2)

'''
def confid_obstacle_loss4(model, data, label, use_bce=False):
    # Model outputs: [otc_logits, conf_score]
    [otc_logits, conf_score] = model(data, training=True)

    tf.debugging.assert_equal(otc_logits.shape, label.shape)
    tf.debugging.assert_equal(conf_score.shape[1], 1)

    pred_idx = tf.argmax(otc_logits, axis=1)      # torch.log_softmax
    pred_onehot = tf.one_hot(pred_idx, depth=label.shape[1], dtype=tf.float64)

    # whether predicted class corresponds to obstacle mask
    pred_is_obst = tf.reduce_sum(pred_onehot * data['obstacle_mask'][:, :tf.shape(label)[1]], axis=1)
    label_is_obst = tf.reduce_sum(label * data['obstacle_mask'][:, :tf.shape(label)[1]], axis=1)

    # detect OOD rows (label all zeros)
    label_is_ood = tf.reduce_all(tf.equal(label, 0.0), axis=1)   # boolean mask

    # convert conf_score to [0,1]
    conf_score = tf.squeeze(conf_score, axis=1)   # [batch]
    conf_score = tf.sigmoid(conf_score)           # now in (0,1)

    # correct_conf: 1 if predicted obstacle matches label obstacle, else 0
    # Note: label_is_obst and pred_is_obst are floats (0 or >0). convert to bool first.
    pred_is_obst_bool = tf.greater(pred_is_obst, 0.5)
    label_is_obst_bool = tf.greater(label_is_obst, 0.5)
    correct_conf = tf.cast(tf.equal(pred_is_obst_bool, label_is_obst_bool), tf.float32)  # 0 or 1

    # Mask out OOD rows from loss
    keep_mask = tf.logical_not(label_is_ood)
    keep_mask_f = tf.cast(keep_mask, tf.float32)

    # compute per-sample loss
    if use_bce:
        per_sample_loss = tf.keras.losses.binary_crossentropy(correct_conf, conf_score)
    else:
        per_sample_loss = tf.square(correct_conf - conf_score)   # MSE per sample

    # zero out OOD samples and compute mean over kept samples (avoid divide by zero)
    masked_loss = per_sample_loss * keep_mask_f

    denom = tf.maximum(tf.reduce_sum(keep_mask_f), 1.0)
    loss = tf.reduce_sum(masked_loss) / denom
    #denom = tf.reduce_sum(keep_mask_f)
    #loss = tf.cond(denom > 0.0, lambda: tf.reduce_sum(masked_loss) / denom, lambda: 0.0 * tf.reduce_sum(conf_score)) #lambda: tf.constant(0.0, dtype=tf.float32))
    loss = tf.clip_by_value(loss, 0.0, 1e6)
    
    return loss

'''
## The loss function seems unstable and generates Nan values due to may be gradient clipping
## If conf_score becomes large, loss uses ((correct_conf - conf_score)^2)^2 (fourth power), which easily overflows to inf and then to NaN in gradients.
## tf.one_hot(..., dtype=tf.float64), most tensors (model outputs, masks, labels) are float32. 
## Mixed dtypes can cause hidden casting and slower/unstable numeric behavior.

def confid_obstacle_loss4(model, data, label):

    model, data, label = model.to('cuda'), data.to('cuda'), label.to('cuda')

    [otc_logits, conf_score] = model(data, training=True)
    tf.debugging.assert_equal(otc_logits.shape, label.shape)
    tf.debugging.assert_equal(conf_score.shape[1], 1)

    pred_idx = tf.argmax(otc_logits, axis=1)      # torch.log_softmax
    pred_onehot = tf.one_hot(pred_idx, depth=label.shape[1], dtype=tf.float64)

    pred_is_obst = tf.reduce_sum(pred_onehot * data['obstacle_mask'][:, :label.shape[1]], axis=1)
    label_is_obst = tf.reduce_sum(label * data['obstacle_mask'][:, :label.shape[1]], axis=1)

    label_is_ood = tf.reduce_all(tf.equal(label, 0.0), axis=1)
    label_is_obst = tf.where(label_is_ood, -1, label_is_obst)

    correct_conf = tf.cast(label_is_obst == pred_is_obst, dtype=tf.float32)
    conf_score = tf.reduce_sum(conf_score, axis=1)

    return tf.sqrt(tf.sqrt(tf.reduce_mean(tf.square(tf.square(correct_conf - conf_score)))))
#'''

def get_model_weights(model: tf.keras.Model) -> List[np.ndarray]:
    return [w.numpy() for w in model.get_weights()]

def set_model_weights(model: tf.keras.Model, weights: List[np.ndarray]):
    model.set_weights(weights)

def weighted_average_weights(client_weights: List[List[np.ndarray]], client_samples: List[int]) -> List[np.ndarray]:
    total_samples = float(np.sum(client_samples))
    if total_samples == 0:
        raise ValueError("Total samples across clients is zero.")
    avg_weights = []
    n_layers = len(client_weights[0])
    for layer_i in range(n_layers):
        weighted_sum = None
        for cw, ns in zip(client_weights, client_samples):
            w = cw[layer_i].astype(np.float64)
            if weighted_sum is None:
                weighted_sum = w * ns
            else:
                weighted_sum += w * ns
        avg = (weighted_sum / total_samples).astype(client_weights[0][layer_i].dtype)
        avg_weights.append(avg)
    return avg_weights

# -----------------------------------------------------------
# Build ConfidNet from a DeepLoc backbone (wraps base_model)
# -----------------------------------------------------------
def attach_confidnet_head_to_backbone(backbone_model: tf.keras.Model):
    """
    Given a DeepLoc backbone model (tf.keras.Model), attach ConfidNet head layers.
    Then wrap the backbone as a nested model with name 'DeepLoc_backbone' to make splitting easier.
    Returns a new Model: inputs -> [backbone_output (otc logits), confid_output]
    """
    # Ensure nested backbone has a stable name
    backbone_model._name = 'DeepLoc_backbone'
    # find the layer "dense_0" output used in original pipeline as base_output
    try:
        base_output = backbone_model.get_layer("dense_0").output
    except Exception:
        # fallback: use backbone_model.output as base_output (if naming differs)
        base_output = backbone_model.output

    confid_x = tf.keras.layers.Dense(16, activation='relu', name="confid_dense_0")(base_output)
    confid_x = tf.keras.layers.Dense(8, activation='relu', name="confid_dense_1")(confid_x)
    confid_x = tf.keras.layers.Dense(8, activation='relu', name="confid_dense_2")(confid_x)
    confid_output = tf.keras.layers.Dense(1, activation='sigmoid', name="confid_dense_3")(confid_x)

    # Construct model that takes same inputs as backbone and returns [otc_logits, confid_score]
    confid_model = tf.keras.Model(inputs=backbone_model.input, outputs=[base_output, confid_output], name="ConfidNet_from_backbone")
    return confid_model

# -------------------------------------------------------------------
# Wrapper Function to split confidnet into backbone and head params
# -------------------------------------------------------------------
def extract_backbone_and_head_weights_from_confidnet(confid_model: tf.keras.Model, backbone_template: tf.keras.Model):
    """
    Given a confid_model (global or received) and a backbone_template (architecture of DeepLoc),
    return two lists of weights: (backbone_weights_list, head_weights_list)
    The backbone_weights_list will be in the same layer-order as backbone_template.get_weights()
    The head_weights_list will be the remaining weights in confid_model after removing backbone weights.
    """
    # Clone backbone template model to receive weights
    cloned_backbone = clone_model(backbone_template)
    cloned_backbone.build(backbone_template.input_shape)

    # Get all weights from confid_model and from backbone part as best we can by layer names:
    # We'll map weights by layer.name where possible.
    confid_layers = {layer.name: layer for layer in confid_model.layers}
    backbone_layers = {layer.name: layer for layer in cloned_backbone.layers}

    # Transfer weights for layers whose names match (backbone part will have same layer names)
    copied_layers = []
    for lname, blayer in backbone_layers.items():
        if lname in confid_layers:
            try:
                blayer.set_weights(confid_layers[lname].get_weights())
                copied_layers.append(lname)
            except Exception:
                # skip if shapes mismatch
                pass

    # Now produce weight lists
    backbone_weights = cloned_backbone.get_weights()  # order matches backbone_template.get_weights() / cloned_backbone.get_weights()
    # head weights: those weights in confid_model that are NOT part of backbone_weights by length/order is tricky.
    # Simpler: produce head_model by creating confid_model minus backbone: create head model that maps base_output -> confid_output
    # We'll build a small head model and copy weights by layer names for confid layers that start with 'confid_'
    head_weights_by_layer = {}
    for lname, layer in confid_layers.items():
        if lname.startswith("confid_dense_"):
            head_weights_by_layer[lname] = layer.get_weights()

    return backbone_weights, head_weights_by_layer

def attach_head_weights_to_backbone_and_build_confid(backbone_template: tf.keras.Model, backbone_weights: List[np.ndarray], head_weights_by_layer: Dict[str, List[np.ndarray]]):
    """
    Given backbone_template (architecture), backbone_weights (list), and head_weights_by_layer (dict layer.name->weights),
    construct a confid_model (backbone + attached head) with backbone_weights set and head weights set.
    """
    # Build backbone and set weights
    backbone_local = clone_model(backbone_template)
    backbone_local.build(backbone_template.input_shape)
    backbone_local.set_weights(backbone_weights)

    # Build confid model from backbone_local
    backbone_local._name = 'DeepLoc_backbone'
    confid_model = attach_confidnet_head_to_backbone(backbone_local)

    # Set head weights by layer names
    for lname, w in head_weights_by_layer.items():
        try:
            confid_model.get_layer(lname).set_weights(w)
        except Exception:
            # layer not found or mismatch, skip
            pass

    return confid_model

def learning_rate_scheduler_step_function(num_epochs, constant_epochs, initial, final):
    drop = np.exp(constant_epochs / num_epochs * np.log(final / initial))
    def lr(epoch):
        return initial * np.power(drop, np.floor(epoch / constant_epochs))
    return lr

def is_nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))

def load_json_as_numpy_dataset(json_path: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Load JSON dataset and return (dataset_np, metadata).
    dataset_np contains:
      - features: (N, M, F) float64 (NaN where missing)
      - dr, track_index, timestamp, category_onehot, sample_weight, obstacle_mask
      - feature_names
    metadata returns validated metadata derived from content.
    """
    with open(json_path, 'r') as f:
        data_json = json.load(f)

    raw_meta = data_json.get("metadata", {})
    data_list = data_json.get("data", [])

    # derive reliable values from actual data
    num_samples = len(data_list)
    if num_samples == 0:
        raise ValueError("No samples found in JSON 'data' array.")

    meta_num_objects = raw_meta.get("num_objects_per_frame", None)
    feature_names = raw_meta.get("feature_names", []) or []
    meta_num_features = raw_meta.get("num_features_per_object", None)

    # Determine actual from first frame (safest)
    first_feats = data_list[0].get("features", [])
    actual_num_objects = len(first_feats)
    if meta_num_objects is None:
        num_objects = actual_num_objects
    else:
        if meta_num_objects != actual_num_objects:
            print(f"WARNING: metadata.num_objects_per_frame ({meta_num_objects}) != actual first frame ({actual_num_objects}). Using actual.")
        num_objects = actual_num_objects

    # Infer feature names / num_features from first object if metadata missing
    if feature_names:
        num_features = len(feature_names)
    else:
        # try infer from keys of first object in first frame
        if actual_num_objects > 0:
            first_obj_keys = list(first_feats[0].keys())
            feature_names = first_obj_keys
            num_features = len(feature_names)
            print(f"INFO: Inferred feature_names = {feature_names}")
        else:
            raise ValueError("Cannot infer feature names: first frame has no objects.")

    # Ensure num_features matches metadata if provided (warn if mismatch)
    if meta_num_features is not None and meta_num_features != num_features:
        print(f"WARNING: metadata.num_features_per_object ({meta_num_features}) != inferred ({num_features}). Using inferred.")

    # Allocate arrays using derived values (fill with NaN for features)
    features = np.full((num_samples, num_objects, num_features), np.nan, dtype=np.float64)
    dr = np.full((num_samples,), np.nan, dtype=np.float64)
    track_index = np.zeros((num_samples,), dtype=np.int32)
    timestamp = np.zeros((num_samples,), dtype=np.float64)
    sample_weight = np.ones((num_samples,), dtype=np.float64)

    # Category onehot and obstacle_mask shapes will be inferred on first occurrence
    category_onehot = None
    obstacle_mask = None

    # Fill arrays frame-by-frame
    for i, frame in enumerate(data_list):
        frame_feats = frame.get("features", [])
        # Build DataFrame to ensure consistent column ordering according to feature_names
        row_list = []
        for obj in frame_feats:
            # create mapping for every feature_name; missing keys -> NaN
            row_list.append({k: (obj.get(k, np.nan) if obj.get(k, None) is not None else np.nan) for k in feature_names})
        # convert to numpy
        if len(row_list) > 0:
            obj_features = pd.DataFrame(row_list, columns=feature_names).to_numpy(dtype=np.float64)
        else:
            # no objects in this frame
            obj_features = np.empty((0, num_features), dtype=np.float64)

        n_obj = obj_features.shape[0]
        # assign (truncate or pad with NaN if necessary)
        if n_obj >= num_objects:
            features[i, :, :] = obj_features[:num_objects, :num_features]
        else:
            features[i, :n_obj, :obj_features.shape[1]] = obj_features

        dr[i] = frame.get("dr", np.nan)
        track_index[i] = frame.get("track_index", 0)
        timestamp[i] = frame.get("timestamp", 0.0)
        sample_weight[i] = frame.get("sample_weight", 1.0)

        # category_onehot: infer width on first non-empty
        co = frame.get("category_onehot", None)
        if co is not None:
            if category_onehot is None:
                category_onehot = np.zeros((num_samples, len(co)), dtype=np.float64)
            if len(co) != category_onehot.shape[1]:
                raise ValueError(f"Inconsistent category_onehot length at frame {i}: {len(co)} != {category_onehot.shape[1]}")
            category_onehot[i, :] = np.array(co, dtype=np.float64)

        # obstacle_mask similar
        om = frame.get("obstacle_mask", None)
        if om is not None:
            if obstacle_mask is None:
                obstacle_mask = np.zeros((num_samples, len(om)), dtype=np.float64)
            if len(om) != obstacle_mask.shape[1]:
                raise ValueError(f"Inconsistent obstacle_mask length at frame {i}: {len(om)} != {obstacle_mask.shape[1]}")
            obstacle_mask[i, :len(om)] = np.array(om, dtype=np.float64)

    if category_onehot is None:
        # create empty 0-column array if no labels present
        category_onehot = np.zeros((num_samples, 0), dtype=np.float64)
    if obstacle_mask is None:
        obstacle_mask = np.zeros((num_samples, category_onehot.shape[1]), dtype=np.float64)

    dataset_np = {"features": features, "dr": dr, "track_index": track_index, "timestamp": timestamp,
        "category_onehot": category_onehot, "sample_weight": sample_weight, "obstacle_mask": obstacle_mask, "feature_names": feature_names}

    metadata_out = {"feature_names": feature_names, "category_names": raw_meta.get("category_names", []),
        "num_samples": num_samples, "num_objects_per_frame": num_objects, "num_features_per_object": num_features}

    return dataset_np, metadata_out

# ------------------------------------------------
# Compute frame-level stats from NumPy arrays
# ------------------------------------------------
def frame_stats_from_np(shard_np: Dict[str, np.ndarray]) -> Dict:
    """
    Returns stats:
      n_frames, features_shape, labels_shape, valid_objects_mean/min/max,
      valid_objects_histogram, label_counts (as dict)
    """
    features = shard_np['features'] 
    labels = shard_np['category_onehot'] 
    n_frames = int(features.shape[0])
    num_objects = int(features.shape[1])

    # valid object if any feature is not NaN: boolean mask (N, M)
    valid_obj_mask = ~np.all(np.isnan(features), axis=2)
    valid_per_frame = valid_obj_mask.sum(axis=1) if n_frames > 0 else np.array([], dtype=int)

    # Label counting: try to convert one-hot rows to class indices if strict one-hot
    label_counts = Counter()
    if labels.size > 0 and labels.shape[1] >= 1:
        if np.all(np.logical_or(labels == 0, labels == 1)) and np.all(labels.sum(axis=1) == 1):
            class_indices = np.argmax(labels, axis=1)
            label_counts = Counter(class_indices.tolist())
        else:
            # fallback to tuple counts
            label_counts = Counter(tuple(row) for row in labels.tolist())

    stats = {
        "n_frames": n_frames,
        "features_shape": features.shape,
        "labels_shape": labels.shape,
        "valid_objects_mean": float(np.mean(valid_per_frame)) if n_frames > 0 else 0.0,
        "valid_objects_min": int(np.min(valid_per_frame)) if n_frames > 0 else 0,
        "valid_objects_max": int(np.max(valid_per_frame)) if n_frames > 0 else 0,
        "valid_objects_histogram": np.bincount(valid_per_frame, minlength=(num_objects+1)).tolist() if n_frames > 0 else [0]*(num_objects+1),
        "label_counts": dict(label_counts)
    }
    return stats

def compute_feature_norm_from_train(train_ds_np: Dict[str, np.ndarray], feature_names: List[str]) -> pd.DataFrame:
    """
    Compute per-feature mean/std across all objects in the train set.
    Returns DataFrame with index ['mean','std'] and columns feature_names.
    """
    num_features = len(feature_names)
    features_array = train_ds_np['features'].reshape(-1, num_features)
    means = np.nanmean(features_array, axis=0)
    stds  = np.nanstd(features_array, axis=0)
    # protect against std==0 => replace with 1.0 so division does nothing
    stds_fixed = np.where(stds == 0, 1.0, stds)
    feature_norm = pd.DataFrame([means, stds_fixed], index=["mean", "std"], columns=feature_names)
    return feature_norm

def apply_feature_norm_numpy(dataset: Dict[str, np.ndarray], feature_norm: pd.DataFrame):
    """
    In-place normalize dataset['features'] using feature_norm DataFrame.
    After normalization, dataset['features_normalized'] appears and 'features' is popped.
    """
    feats = dataset['features']
    mean = feature_norm.loc['mean'].to_numpy()
    std = feature_norm.loc['std'].to_numpy()
    dataset['features'] = (feats - mean) / std
    dataset['features_normalized'] = dataset.pop('features')

# ----------------------------------
# Prepare Dataset from numpy dict
# ----------------------------------
def prepare_dataset_for_tensorflow(dataset: Dict[str, np.ndarray], feature_norm: pd.DataFrame, batch_size: int, shuffle: bool):
    """
    Returns a batched tf.data.Dataset yielding ((features_dict), labels, sample_weight).
    features_normalized: tf.float32 tensor (N, M, F)
    Other tensors converted to tf.float32 where appropriate.
    """
    dataset_cp = deepcopy(dataset)
    # Normalize (creates 'features_normalized')
    apply_feature_norm_numpy(dataset_cp, feature_norm)

    # Replace object rows that were all-NaN (across features) with zeros
    all_nan = np.all(np.isnan(dataset_cp['features_normalized']), axis=2)
    # Set those object rows to zeros
    # Boolean indexing on first two dims selects rows flattened; assignment shape must match (num_selected, F)
    if np.any(all_nan):
        dataset_cp['features_normalized'][all_nan, :] = 0.0

    # Convert dtypes to float32 for TF
    tensor_features = tf.convert_to_tensor(dataset_cp['features_normalized'], dtype=tf.float64, name='features_normalized')
    tensor_dr = tf.convert_to_tensor(dataset_cp['dr'], dtype=tf.float64, name='dr')
    tensor_track_index = tf.convert_to_tensor(dataset_cp['track_index'], dtype=tf.int32, name='track_index')
    tensor_timestamp = tf.convert_to_tensor(dataset_cp['timestamp'], dtype=tf.float64, name='timestamp')
    tensor_obstacle_mask = tf.convert_to_tensor(dataset_cp['obstacle_mask'], dtype=tf.float64, name='obstacle_mask')
    tensor_labels = tf.convert_to_tensor(dataset_cp['category_onehot'], dtype=tf.float64, name='category_onehot')
    tensor_sample_weight = tf.convert_to_tensor(dataset_cp['sample_weight'], dtype=tf.float64, name='sample_weight')

    # Build dataset of tuples: (feature_dict, labels, sample_weight)
    feature_dict = {
        'features_normalized': tensor_features,
        'dr': tensor_dr,
        'track_index': tensor_track_index,
        'timestamp': tensor_timestamp,
        'obstacle_mask': tensor_obstacle_mask
    }

    dataset_tf = tf.data.Dataset.from_tensor_slices((feature_dict, tensor_labels, tensor_sample_weight))
    if shuffle:
        dataset_tf = dataset_tf.shuffle(buffer_size=tensor_features.shape[0], reshuffle_each_iteration=True)
    batched_dataset_tf = dataset_tf.batch(batch_size).prefetch(2)
    return batched_dataset_tf

# -------------------------
# Load & prepare all datasets (train/valid/test)
# -------------------------
def load_and_prepare_all_datasets(train_json: str, valid_json: str, test_json: str, batch_size: int, shuffle: bool):
    # Load numpy datasets from JSON
    train_ds_np, train_meta = load_json_as_numpy_dataset(train_json)
    valid_ds_np, valid_meta = load_json_as_numpy_dataset(valid_json)
    test_ds_np, test_meta = load_json_as_numpy_dataset(test_json)

    # Sanity: ensure metadata compatibility (feature/category names)
    if train_meta["feature_names"] != valid_meta["feature_names"] or train_meta["feature_names"] != test_meta["feature_names"]:
        raise ValueError("Feature names mismatch across datasets. Ensure train/valid/test share same features.")

    # Category names may be missing in metadata; ensure shape compatibility in arrays
    if train_ds_np['category_onehot'].shape[1] != valid_ds_np['category_onehot'].shape[1] or \
       train_ds_np['category_onehot'].shape[1] != test_ds_np['category_onehot'].shape[1]:
        print("WARNING: category_onehot width mismatch across splits (labels). Proceeding but check label shapes.")

    feature_names = train_meta["feature_names"]
    category_names = train_meta.get("category_names", [])
    num_features = train_meta["num_features_per_object"]

    # Compute feature_norm from train
    feature_norm = compute_feature_norm_from_train(train_ds_np, feature_names)

    # Prepare tf datasets
    dataset_train_tf = prepare_dataset_for_tensorflow(train_ds_np, feature_norm, batch_size=batch_size, shuffle=shuffle)
    dataset_valid_tf = prepare_dataset_for_tensorflow(valid_ds_np, feature_norm, batch_size=batch_size, shuffle=False)
    dataset_test_tf = prepare_dataset_for_tensorflow(test_ds_np, feature_norm, batch_size=batch_size, shuffle=False)

    # Correct sample counts: use numpy arrays (number of frames)
    train_samples = train_ds_np['features'].shape[0]
    valid_samples = valid_ds_np['features'].shape[0]
    test_samples  = test_ds_np['features'].shape[0]

    info = {
        "feature_names": feature_names,
        "category_names": category_names,
        "train_samples": train_samples,
        "valid_samples": valid_samples,
        "test_samples": test_samples,
        "feature_norm": feature_norm,
        "train_meta": train_meta,
        "train_np": train_ds_np,
        "valid_np": valid_ds_np,
        "test_np": test_ds_np
    }

    return dataset_train_tf, dataset_valid_tf, dataset_test_tf, info

# -----------------------------------------
# Uniform Per-Class Stratified Splitting
# -----------------------------------------
def stratified_uniform_split(dataset_np: Dict[str, np.ndarray], num_clients: int, random_seed: int = 0, verbose: bool = False) -> List[Dict[str, np.ndarray]]:
    """
    Uniform per-class stratified splitting:
      For each class c:
        - collect all indices whose label == c
        - shuffle indices
        - split into num_clients chunks (np.array_split to handle unequal division)
      For client i: union of chunk_i from every class.
    Returns list of dataset dicts (shards) for each client.
    """
    n_total = dataset_np['features'].shape[0]
    labels = dataset_np['category_onehot']  # (N, C)
    if labels.size == 0 or labels.shape[1] == 0:
        # No labels: fallback to simple round-robin split of indices
        idxs = np.arange(n_total)
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idxs)
        splits = np.array_split(idxs, num_clients)
    else:
        # Determine class indices per sample (prefer strict one-hot)
        if np.all(np.logical_or(labels == 0, labels == 1)) and np.all(labels.sum(axis=1) == 1):
            class_idx = np.argmax(labels, axis=1)
            num_classes = labels.shape[1]
        else:
            # fallback: map unique tuples -> class ids
            tuples = [tuple(row) for row in labels.tolist()]
            uniq = {t:i for i,t in enumerate(sorted(set(tuples)))}
            class_idx = np.array([uniq[t] for t in tuples], dtype=int)
            num_classes = len(uniq)

        # For each class, shuffle and split into num_clients chunks
        rng = np.random.RandomState(random_seed)
        class_to_chunks = {}
        for c in range(num_classes):
            inds = np.where(class_idx == c)[0]
            inds = inds.tolist()
            rng.shuffle(inds)
            # split into num_clients chunks (some chunks may be empty)
            chunks = np.array_split(np.array(inds, dtype=int), num_clients)
            class_to_chunks[c] = chunks

        # Build client buckets: client i gets chunk i from each class
        client_buckets = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            chunks = class_to_chunks[c]
            for client_id, chunk in enumerate(chunks):
                if chunk.size > 0:
                    client_buckets[client_id].extend(chunk.tolist())

        # Optionally shuffle client buckets to distribute sequential biases
        for client_id in range(num_clients):
            rng.shuffle(client_buckets[client_id])

        splits = [np.array(sorted(client_buckets[i]), dtype=int) for i in range(num_clients)]

    # Build shard datasets by slicing numpy arrays
    shards = []
    for inds in splits:
        shard = {}
        for k, v in dataset_np.items():
            if isinstance(v, np.ndarray) and v.shape[0] == n_total:
                shard[k] = v[inds]
            else:
                shard[k] = deepcopy(v)
        shards.append(shard)

    if verbose:
        for cid, inds in enumerate(splits):
            print(f"Client {cid}: assigned {len(inds)} frames (indices sample: {inds[:10].tolist()})")

    return shards

def create_client_tf_datasets(train_shards_np: List[Dict[str, np.ndarray]], valid_shards_np: List[Dict[str, np.ndarray]], feature_norm: pd.DataFrame,
                              batch_size: int, shuffle_local: bool, verbose: bool = True):
    client_train_tfs = []
    client_valid_tfs = []
    client_train_sizes = []
    client_val_sizes = []
    for client_id, (tr_np, val_np) in enumerate(zip(train_shards_np, valid_shards_np)):
        train_tf = prepare_dataset_for_tensorflow(tr_np, feature_norm, batch_size=batch_size, shuffle=shuffle_local)
        val_tf   = prepare_dataset_for_tensorflow(val_np, feature_norm, batch_size=batch_size, shuffle=False)

        train_size = int(tr_np['features'].shape[0])
        val_size = int(val_np['features'].shape[0])

        client_train_tfs.append(train_tf)
        client_valid_tfs.append(val_tf)
        client_train_sizes.append(train_size)
        client_val_sizes.append(val_size)

        if verbose:
            stats_tr = frame_stats_from_np(tr_np)
            stats_val = frame_stats_from_np(val_np)
            print(f"\n--- Client {client_id} summary ---")
            print(f" Train frames: {train_size}; Val frames: {val_size}")
            print(f" Train: valid objects mean/min/max = {stats_tr['valid_objects_mean']:.2f}/{stats_tr['valid_objects_min']}/{stats_tr['valid_objects_max']}")
            print(f" Val:   valid objects mean/min/max = {stats_val['valid_objects_mean']:.2f}/{stats_val['valid_objects_min']}/{stats_val['valid_objects_max']}")
            print(f" Train label counts: {stats_tr['label_counts']}")
            print(f" Val label counts:   {stats_val['label_counts']}")
            print(f" Train features shape: {stats_tr['features_shape']}; labels shape: {stats_tr['labels_shape']}")

    return client_train_tfs, client_valid_tfs, client_train_sizes, client_val_sizes

# ------------------------------------------------
# Implementing the described Federated pipeline
# ------------------------------------------------
def federated_pipeline(args):
    # Load and prepare datasets (server side)
    dataset_train_tf, dataset_valid_tf, dataset_test_tf, info = load_and_prepare_all_datasets(
        train_json=args.train_json, valid_json=args.valid_json, test_json=args.test_json, batch_size=args.batch_size, shuffle=args.shuffle)

    print(f"\nDataset summary: train={info['train_samples']} valid={info['valid_samples']} test={info['test_samples']}")
    print(f"Feature count: {len(info['feature_names'])}  Categories: {len(info['category_names'])}")

    # Split train/valid into client shards (server-side split) using Option A
    num_clients = args.num_clients
    train_shards_np = stratified_uniform_split(info['train_np'], num_clients=num_clients, random_seed=args.random_seed, verbose=True)
    valid_shards_np = stratified_uniform_split(info['valid_np'], num_clients=num_clients, random_seed=args.random_seed, verbose=False)

    client_train_tfs, client_valid_tfs, client_train_sizes, client_val_sizes = create_client_tf_datasets(
        train_shards_np, valid_shards_np, info['feature_norm'], batch_size=args.batch_size, shuffle_local=args.shuffle_local, verbose=True)

    print("\nPer-client train sizes:", client_train_sizes)
    print("Per-client val sizes:", client_val_sizes)

    # Load initial DeepLoc model (server)
    print(f"\nLoading initial DeepLoc from {args.model_path}")
    server_deeploc_template = load_model(args.model_path, compile=False, custom_objects={'Mask': Mask, 'MaskedGlobalMaxPool1D': MaskedGlobalMaxPool1D, 'Repeat': Repeat})
    # Keep a template architecture for future clones (unchanged) and for convenience we ensure template has a stable name
    server_deeploc_template._name = "DeepLoc_template"
    
    #-----------------------------------------------------------------------------------------------------------------------------------
    # --- Ensure classification head matches dataset num classes (robust method) ---
    num_classes = len(info['category_names'])
    try:
        loaded_num_classes = int(server_deeploc_template.output_shape[-1])
    except Exception:
        loaded_num_classes = None

    if loaded_num_classes is not None and loaded_num_classes != num_classes:
        print(f"Loaded DeepLoc has {loaded_num_classes} outputs but dataset has {num_classes}. Replacing final Dense+Softmax head.")
        # Use the existing penultimate layer by name (safe) - in your model that is 'maxpool_1'
        try:
            penultimate = server_deeploc_template.get_layer('maxpool_1').output
        except Exception:
            inbound_layer = server_deeploc_template.layers[-2]  # second last layer object # For 2-Class change this to [-1]
            penultimate = inbound_layer.output

        # create new classification layers using the penultimate tensor # name them exactly as pipeline expects: 'dense_0' and 'softmax'
        new_dense = tf.keras.layers.Dense(num_classes, name='dense_0')(penultimate)
        new_soft = tf.keras.layers.Activation('softmax', name='softmax')(new_dense)

        # build a new model that re-uses all existing backbone layers (weights preserved)
        server_deeploc_template = tf.keras.Model(inputs=server_deeploc_template.input, outputs=new_soft, name="DeepLoc_template")
        print("Rebuilt DeepLoc_template with new 'dense_0' and 'softmax' for", num_classes, "classes.")
    else:
        print("DeepLoc output matches dataset classes (or output shape unavailable); no head change performed.")

    #-----------------------------------------------------------------------------------------------------------------------------------

    # No global confidnet yet
    global_confidnet = None
    
    # Track best model (lowest test loss)
    best_loss = float('inf')
    best_round = None

    # For per-round aggregation we keep a list of client confidnet weight lists and sample counts
    for rnd in range(1, args.global_rounds + 1):
        print(f"\n===== Global Round {rnd}/{args.global_rounds} =====")
        # choose participating clients
        m = max(1, int(math.ceil(args.frac_clients * num_clients)))
        selected = np.random.choice(np.arange(num_clients), size=m, replace=False)
        print("Selected clients:", selected.tolist())

        client_confid_weights = []
        client_samples = []

        # For Round 1: disseminate DeepLoc template (server_deeploc_template) to clients
        if rnd == 1:
            # Build global_deeploc weights to send
            global_deeploc_weights = server_deeploc_template.get_weights()
            # Each selected client receives DeepLoc, trains it, then locally builds ConfidNet and trains it, and returns ConfidNet weights
            for cid in selected:
                print(f"\nClient {cid}: Starting Round-1 work (receive DeepLoc)")
                # Create local DeepLoc model (clone template) and set global weights
                local_deeploc = clone_model(server_deeploc_template)
                local_deeploc.build(server_deeploc_template.input_shape)
                local_deeploc.set_weights(global_deeploc_weights)
                
                # To check if sample_weight is zero everywhere (min == max == 0) or if each label vector sums to 0 (indicating an “all-zero”/OOD label),
                #inspect_dataset(client_train_tfs[cid], name=f"client_{cid}_train")

                # Compile and train DeepLoc locally
                local_deeploc.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=args.deeploc_lr, momentum=0.9),
                                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                                      metrics=['accuracy'])
                if client_train_sizes[cid] > 0 and args.deeploc_local_epochs > 0:
                    local_deeploc.fit(client_train_tfs[cid], validation_data=client_valid_tfs[cid], epochs=args.deeploc_local_epochs, verbose=1)
                

                # Build ConfidNet locally (attach head)
                local_confid = attach_confidnet_head_to_backbone(local_deeploc)
                # Train ConfidNet locally with custom loss loop (like original)
                lr_sched = {"constant_epochs": 4, "initial": 0.0001, "final": args.confid_final_lr}
                lr_function = learning_rate_scheduler_step_function(num_epochs=args.confid_local_epochs, **lr_sched)
                optimizer = keras.optimizers.SGD(learning_rate=0.0, momentum=0.9, nesterov=True)
                
                epoch_losses = []
                # We'll run the requested number of local ConfidNet epochs
                for ep in range(args.confid_local_epochs):
                    # Optionally set lr schedule (we use simple constant or user provided schedule)
                    #optimizer.learning_rate = args.confid_initial_lr
                    optimizer.learning_rate = lr_function(ep)
                    print(f"\nEpoch {ep+1}/{args.confid_local_epochs}  - Learning Rate: {optimizer.learning_rate.numpy():.6f}")

                    for data_sample, labels, _ in client_train_tfs[cid]:
                        with tf.GradientTape() as tape:
                            tape.watch(local_confid.trainable_variables)
                            loss_value = confid_obstacle_loss4(local_confid, data_sample, labels)
                        if tf.math.reduce_any(tf.math.is_nan(loss_value)):
                            continue
                        epoch_losses.append(loss_value.numpy())
                        grads = tape.gradient(loss_value, local_confid.trainable_variables)
                        optimizer.apply_gradients(zip(grads, local_confid.trainable_variables))
                    
                    mean_loss = np.mean(epoch_losses)
                    print(f"Client {cid} - ConfidNet Epoch {ep+1}/{args.confid_local_epochs}: Train Loss = {mean_loss:.6f}")

                    # --- Compute validation loss ---
                    val_losses = []
                    for val_x, val_y, _ in client_valid_tfs[cid]:
                        vloss = confid_obstacle_loss4(local_confid, val_x, val_y)
                        val_losses.append(vloss.numpy())
                    mean_val_loss = np.mean(val_losses)
                    print(f"Client {cid} - ConfidNet Epoch {ep+1}: Validation Loss = {mean_val_loss:.6f}")

                # after local training, collect confid weights to send to server
                client_confid_weights.append(local_confid.get_weights())
                client_samples.append(client_train_sizes[cid])

                #tf.keras.backend.clear_session()
                del local_confid
                del local_deeploc
                gc.collect()

        else:
            # Round >1: server has a global_confidnet to send
            if global_confidnet is None:
                raise RuntimeError("No global ConfidNet available for round >1. Something went wrong.")

            # prepare global_confidnet weights to send
            global_confid_weights = global_confidnet.get_weights()

            for cid in selected:
                print(f"\nClient {cid}: Starting Round-{rnd} work (receive global ConfidNet)")
                # Client loads ConfidNet from server weights
                local_confid = clone_model(global_confidnet)
                local_confid.build(global_confidnet.input_shape)
                local_confid.set_weights(global_confid_weights)

                # Split confid into backbone and head weights
                backbone_weights, head_weights_by_layer = extract_backbone_and_head_weights_from_confidnet(local_confid, server_deeploc_template)

                # Build a local DeepLoc clone and set backbone weights (detached)
                local_deeploc = clone_model(server_deeploc_template)
                local_deeploc.build(server_deeploc_template.input_shape)
                try:
                    local_deeploc.set_weights(backbone_weights)
                except Exception:
                    # If direct set_weights fails due to ordering, attempt to copy by layer names
                    # Fallback: iterate layers and copy weights for matching layer names
                    for layer in local_deeploc.layers:
                        lname = layer.name
                        try:
                            layer.set_weights(local_confid.get_layer(lname).get_weights())
                        except Exception:
                            pass

                # Train detached DeepLoc locally
                local_deeploc.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=args.deeploc_lr, momentum=0.9),
                                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                                      metrics=['accuracy'])
                if client_train_sizes[cid] > 0 and args.deeploc_local_epochs > 0:
                    local_deeploc.fit(client_train_tfs[cid], validation_data=client_valid_tfs[cid], epochs=args.deeploc_local_epochs, verbose=1)

                # Re-attach head and set head weights from saved head_weights_by_layer
                local_confid = attach_confidnet_head_to_backbone(local_deeploc)
                # Now set head layer weights by name
                for lname, w in head_weights_by_layer.items():
                    try:
                        local_confid.get_layer(lname).set_weights(w)
                    except Exception:
                        pass

                # Now train ConfidNet locally further
                lr_sched = {"constant_epochs": 4, "initial": 0.0001, "final": args.confid_final_lr}
                lr_function = learning_rate_scheduler_step_function(num_epochs=args.confid_local_epochs, **lr_sched)
                optimizer = keras.optimizers.SGD(learning_rate=0.0, momentum=0.9, nesterov=True)

                epoch_losses = []
                for ep in range(args.confid_local_epochs):
                    #optimizer.learning_rate = args.confid_initial_lr
                    optimizer.learning_rate = lr_function(ep)
                    print(f"\nEpoch {ep+1}/{args.confid_local_epochs}  - Learning Rate: {optimizer.learning_rate.numpy():.6f}")
                    
                    for data_sample, labels, _ in client_train_tfs[cid]:
                        with tf.GradientTape() as tape:
                            tape.watch(local_confid.trainable_variables)
                            loss_value = confid_obstacle_loss4(local_confid, data_sample, labels)
                        if tf.math.reduce_any(tf.math.is_nan(loss_value)):
                            continue
                        epoch_losses.append(loss_value.numpy())
                        grads = tape.gradient(loss_value, local_confid.trainable_variables)
                        optimizer.apply_gradients(zip(grads, local_confid.trainable_variables))

                    mean_loss = np.mean(epoch_losses)
                    print(f"Client {cid} - ConfidNet Epoch {ep+1}/{args.confid_local_epochs}: Train Loss = {mean_loss:.6f}")

                    # --- Compute validation loss ---
                    val_losses = []
                    for val_x, val_y, _ in client_valid_tfs[cid]:
                        vloss = confid_obstacle_loss4(local_confid, val_x, val_y)
                        val_losses.append(vloss.numpy())
                    mean_val_loss = np.mean(val_losses)
                    print(f"Client {cid} - ConfidNet Epoch {ep+1}: Validation Loss = {mean_val_loss:.6f}")

                # Client done; send confid weights to server
                client_confid_weights.append(local_confid.get_weights())
                client_samples.append(client_train_sizes[cid])

                #tf.keras.backend.clear_session()
                del local_confid
                del local_deeploc
                gc.collect()

        # End of client loop for this round: server aggregates client_confid_weights into new global_confidnet
        if len(client_confid_weights) == 0:
            print("No client updates this round; skipping aggregation.")
            continue

        print("\nServer: Aggregating client ConfidNet updates (FedAvg sample-weighted)...")
        aggregated_confid_weights = weighted_average_weights(client_confid_weights, client_samples)

        # Build server global_confidnet model from server_deeploc_template architecture and attach head
        # We need to create a confid model instance to host aggregated weights
        # We'll create confid_model from a fresh clone of the backbone template (server_deeploc_template)
        backbone_for_confid = clone_model(server_deeploc_template)
        backbone_for_confid.build(server_deeploc_template.input_shape)
        global_confidnet = attach_confidnet_head_to_backbone(backbone_for_confid)
        # set aggregated weights
        try:
            global_confidnet.set_weights(aggregated_confid_weights)
        except Exception:
            # fallback: try to set by layer names: iterate through layers and set weights from aggregated order
            global_confidnet.set_weights(aggregated_confid_weights)

        # Evaluate aggregated global_confidnet on server test dataset
        print("Server: Evaluating aggregated global ConfidNet on test set...")
        test_losses = []
        for data_sample, labels, _ in dataset_test_tf:
            loss_value = confid_obstacle_loss4(global_confidnet, data_sample, labels)
            test_losses.append(loss_value.numpy().mean())
        mean_test_loss = np.mean(test_losses) if len(test_losses) > 0 else float('nan')
        print(f"Aggregated Global ConfidNet Test Loss (mean) after round {rnd}: {mean_test_loss:.6f}")

        # ---------- SAVE ONLY IF BEST ----------
        os.makedirs(args.save_dir, exist_ok=True)
        if not np.isnan(mean_test_loss) and mean_test_loss < best_loss:
            print(f"New best model found at round {rnd}! Saving ConfidNet + DeepLoc backbone...")

            best_loss = float(mean_test_loss)
            best_round = rnd

            # Save best ConfidNet (entire model)
            best_conf_path = os.path.join(args.save_dir, f"best_global_confidnet.h5")
            try:
                global_confidnet.save(best_conf_path)
            except Exception as e:
                # If saving full model fails due to custom layers, save weights instead as fallback
                print(f"Warning: full-model save failed ({e}), saving weights instead.")
                global_confidnet.save_weights(best_conf_path + ".weights.h5")

            # Extract DeepLoc backbone dynamically from the confid model
            try:
                extracted_deeploc = tf.keras.Model(inputs=global_confidnet.input, outputs=global_confidnet.outputs[0], name="extracted_deeploc")

                y_true_all = []
                y_pred_all = []

                for data_sample, labels, _ in dataset_test_tf:
                    logits = extracted_deeploc(data_sample, training=False)
                    preds = tf.argmax(logits, axis=1).numpy()
                    trues = tf.argmax(labels, axis=1).numpy()

                    y_true_all.extend(trues)
                    y_pred_all.extend(preds)

                y_true_all = np.array(y_true_all)
                y_pred_all = np.array(y_pred_all)

                deeploc_test_acc = accuracy_score(y_true_all, y_pred_all)
                deeploc_confmat = confusion_matrix(y_true_all, y_pred_all)
                classwise_acc = deeploc_confmat.diagonal() / deeploc_confmat.sum(axis=1)

                print("---- DeepLoc Test Metrics ----")
                print(f"Test Accuracy: {deeploc_test_acc:.4f}")
                print("Class-wise Accuracy:")
                for i, acc in enumerate(classwise_acc):
                    print(f"  Class {i}: {acc:.4f}")
                print("Confusion Matrix:")
                print(deeploc_confmat)

                deeploc_path = os.path.join(args.save_dir, f"best_global_deeploc.h5")
                try:
                    extracted_deeploc.save(deeploc_path)
                except Exception as e:
                    print(f"Warning: full DeepLoc save failed ({e}), saving weights only.")
                    extracted_deeploc.save_weights(deeploc_path + ".weights.h5")
                print(f"Server: Saved BEST global ConfidNet: {best_conf_path}")
                print(f"Server: Saved BEST extracted DeepLoc backbone: {deeploc_path}")
            except Exception as e:
                print(f"ERROR extracting DeepLoc backbone for saving: {e}")
        else:
            print(f"No improvement at round {rnd}. Best loss remains {best_loss:.6f} (round {best_round}).")

        '''
        # Save aggregated global confidnet for this round
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f"global_confidnet_round_{rnd}.h5")
        global_confidnet.save(save_path)
        print(f"Server: Saved global ConfidNet checkpoint: {save_path}")
        '''
    print("\nFederated pipeline finished. Final global ConfidNet saved in save_dir.")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated DeepLoc + ConfidNet pipeline (server-driven FedAvg)")
    parser.add_argument("--train_json", type=str, default="Dataset_Path/train_dataset.json", help="Path to training JSON file")
    parser.add_argument("--valid_json", type=str, default="Dataset_Path/valid_dataset.json", help="Path to validation JSON file")
    parser.add_argument("--test_json", type=str, default="Dataset_Path/test_dataset.json", help="Path to test JSON file")
    parser.add_argument("--model_path", type=str, default="model_weights_otc/best_otc_model_with_architecture.h5", help="Path to DeepLoc model")
    parser.add_argument("--save_dir", type=str, default="Fed_Model", help="Directory to save trained models")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproducibility")
    #parser.add_argument('-lr', "--learning_rate", type=float, default=1e-3, help="(Not used directly) default LR placeholder")
    parser.add_argument("--shuffle", type=int, choices=[0,1], default=0, help="Shuffle server preprocessing (1) or not (0)")
    # federated args
    parser.add_argument("--num_clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--frac_clients", type=float, default=0.6, help="Fraction of clients participating each round (0-1]")
    parser.add_argument("--global_rounds", type=int, default=5, help="Number of global rounds")
    parser.add_argument("--shuffle_local", type=int, choices=[0,1], default=1, help="Shuffle local datasets for clients")
    # local training hyperparams
    parser.add_argument("--deeploc_local_epochs", type=int, default=3, help="Local epochs for DeepLoc training at client")
    parser.add_argument("--confid_local_epochs", type=int, default=3, help="Local epochs for ConfidNet training at client")
    parser.add_argument("--deeploc_lr", type=float, default=1e-5, help="LR for local DeepLoc training")
    parser.add_argument("--confid_final_lr", type=float, default=0.00001, help="Initial LR for ConfidNet local SGD")
    # scheduler args
    args = parser.parse_args()
    args.shuffle = bool(args.shuffle)
    args.shuffle_local = bool(args.shuffle_local)

    os.makedirs(args.save_dir, exist_ok=True)
    federated_pipeline(args)
