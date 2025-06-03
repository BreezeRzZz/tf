import torch
import torch.nn as nn
import numpy as np
import pickle
import random
import os
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from DF_model_pytorch import DF_PyTorch # Assuming DF_model_pytorch.py is in the same directory or accessible

# --- Configuration ---
DATA_PATH = '../../dataset/ts_mon.pkl'  # Path to your single pickle file
MODEL_PATH = './trained_model/Triplet_Model_PyTorch_epoch6.pth' # Path to your trained PyTorch model
MAX_LEN = 5000
EMBEDDING_SIZE = 64 # Should match your model's output embedding size
N_SHOT_LIST = [1, 5] # Number of examples for n-shot
SIZE_OF_PROBLEM_LIST = [95] # Number of websites to include in each evaluation run (max is num_classes in test set)
NUM_EVAL_RUNS = 10 # Number of times to repeat the evaluation for averaging
TYPE_EXP = 'N-MEV' # 'N-MEV' for N-Shot Mean of Embedded Vectors, or 'RAW' for individual embeddings

def load_test_data(data_path, max_len):
    """Loads test data (last 40 traces per site) from the pickle file."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    site_dict_test = defaultdict(list)
    
    for site_id_orig, traces in data.items():
        # Use the last 40 traces for testing
        traces_for_testing = traces[160:] 
        if not traces_for_testing: # Skip if a site has less than 160 traces
            print(f"Site {site_id_orig} has less than 160 traces, skipping for testing.")
            continue

        for trace_data in traces_for_testing:
            if isinstance(trace_data, np.ndarray):
                trace = trace_data
            elif isinstance(trace_data, list):
                trace = np.array(trace_data)
            else:
                try:
                    trace = np.array(list(trace_data))
                except Exception as e:
                    print(f"Error converting trace_data for site {site_id_orig} (test): {e}")
                    continue

            if trace.ndim == 1:
                trace = trace[:max_len]
                trace = np.pad(trace, (0, max_len - len(trace)), 'constant')
            elif trace.ndim == 2 and trace.shape[1] == 1:
                trace = trace[:max_len, :]
                if trace.shape[0] < max_len:
                    padding = np.zeros((max_len - trace.shape[0], 1))
                    trace = np.vstack((trace, padding))
            else:
                print(f"Skipping trace with unexpected shape: {trace.shape} for site {site_id_orig} (test)")
                continue
            
            site_dict_test[site_id_orig].append(trace.reshape(1, max_len)) # Reshape for Conv1d (C_in=1)

    return site_dict_test

def get_embeddings(model, device, data_dict):
    """Generates embeddings for the given data using the PyTorch model."""
    model.eval()
    embeddings_dict = defaultdict(list)
    with torch.no_grad():
        for site_id, traces in data_dict.items():
            if not traces:
                continue
            # Stack traces for batch processing if possible, or process one by one
            # Assuming traces are already correctly shaped (1, MAX_LEN) and then (N, 1, MAX_LEN) for model
            
            batch_traces = []
            for trace_np in traces:
                 # Ensure trace_np is (1, MAX_LEN), then add channel dim: (1, 1, MAX_LEN)
                if trace_np.ndim == 2 and trace_np.shape[0] == 1: # (1, MAX_LEN)
                    trace_tensor = torch.from_numpy(trace_np).float().unsqueeze(1).to(device) # (1, 1, MAX_LEN)
                else:
                    print(f"Unexpected trace shape for site {site_id}: {trace_np.shape}")
                    continue
                batch_traces.append(trace_tensor)
            
            if not batch_traces:
                continue

            # It's often better to process a batch if memory allows
            # For simplicity here, let's process one by one if stacking is complex
            # Or, if all traces for a site can be batched:
            try:
                site_traces_tensor = torch.cat(batch_traces, dim=0) # (num_traces_for_site, 1, MAX_LEN)
                site_embeddings = model(site_traces_tensor).cpu().numpy()
                embeddings_dict[site_id] = list(site_embeddings) # Store as list of np arrays
            except RuntimeError as e:
                print(f"RuntimeError during embedding generation for site {site_id}: {e}. Processing one by one.")
                # Fallback: process one by one if batching fails (e.g. due to inconsistent trace lengths if not padded)
                for trace_tensor_single in batch_traces:
                    try:
                        embedding = model(trace_tensor_single).cpu().numpy()
                        embeddings_dict[site_id].append(embedding[0]) # model output is (1, embed_dim)
                    except Exception as ex_single:
                        print(f"Error processing single trace for site {site_id}: {ex_single}")      
                        
    return embeddings_dict

def create_signature_and_test_sets(site_embeddings_dict, n_shot, num_sites_for_problem):
    """Creates signature and test sets from embeddings for a specific evaluation run."""
    
    available_sites = list(site_embeddings_dict.keys())
    if len(available_sites) < num_sites_for_problem:
        print(f"Warning: Requested {num_sites_for_problem} sites, but only {len(available_sites)} available. Using all available.")
        num_sites_for_problem = len(available_sites)
    
    selected_sites = random.sample(available_sites, num_sites_for_problem)

    signature_vectors = []
    signature_labels = []
    test_vectors = []
    test_labels = []

    for site_id in selected_sites:
        embeddings = site_embeddings_dict[site_id]
        if len(embeddings) < n_shot:
            print(f"Warning: Site {site_id} has only {len(embeddings)} samples, less than n_shot={n_shot}. Skipping this site for this run.")
            continue
        
        random.shuffle(embeddings)
        
        # Create signature set
        current_signatures = embeddings[:n_shot]
        if TYPE_EXP == "N-MEV":
            mean_signature = np.mean(np.array(current_signatures), axis=0)
            signature_vectors.append(mean_signature)
            signature_labels.append(site_id)
        else: # RAW embeddings
            for sig_emb in current_signatures:
                signature_vectors.append(sig_emb)
                signature_labels.append(site_id)

        # Create test set (remaining samples)
        current_tests = embeddings[n_shot:]
        if not current_tests:
            # print(f"Warning: Site {site_id} has no remaining samples for testing after taking {n_shot} for signature.")
            pass # It's possible if total samples = n_shot

        for test_emb in current_tests:
            test_vectors.append(test_emb)
            test_labels.append(site_id)
            
    return np.array(signature_vectors), np.array(signature_labels), np.array(test_vectors), np.array(test_labels)

def knn_accuracy(X_train, y_train, X_test, y_test, n_neighbors_knn):
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Warning: Empty train or test set for kNN. Returning 0 accuracy.")
        return 0.0, 0.0, 0.0
    
    # Ensure n_neighbors_knn is not greater than number of training samples
    actual_n_neighbors = min(n_neighbors_knn, X_train.shape[0])
    if actual_n_neighbors == 0: # Should not happen if X_train is not empty
        return 0.0, 0.0, 0.0

    knn = KNeighborsClassifier(n_neighbors=actual_n_neighbors, weights='distance', p=2, metric='cosine', algorithm='brute') 
    knn.fit(X_train, y_train)

    # Top-1 Accuracy
    y_pred_top1 = knn.predict(X_test)
    acc_knn_top1 = accuracy_score(y_test, y_pred_top1)

    # Top-2 and Top-5 Accuracy
    probas = knn.predict_proba(X_test)
    acc_knn_top2 = 0
    acc_knn_top5 = 0

    if X_train.shape[0] >= 2: # Need at least 2 classes for top-2
        best_2 = np.argsort(probas, axis=1)[:,-2:]
        for i in range(len(y_test)):
            if y_test[i] in knn.classes_[best_2[i]]:
                acc_knn_top2 += 1
        acc_knn_top2 /= len(y_test)
    else: # If only 1 class in training, top-2 is same as top-1
        acc_knn_top2 = acc_knn_top1

    if X_train.shape[0] >= 5: # Need at least 5 classes for top-5
        best_5 = np.argsort(probas, axis=1)[:,-5:]
        for i in range(len(y_test)):
            if y_test[i] in knn.classes_[best_5[i]]:
                acc_knn_top5 += 1
        acc_knn_top5 /= len(y_test)
    elif X_train.shape[0] > 0 : # If 1 to 4 classes, top-5 is same as top-1 (or top-N if N < 5)
        # More accurately, it's top-min(5, num_classes_in_train)
        # For simplicity, if less than 5 classes, we can consider it as top-N where N is num_classes
        # which means if the true class is among any of the predicted (up to N), it's correct.
        # This is effectively the same as top-1 if only 1 class, or if prediction is always within the small set.
        # A simpler approach for now: if less than 5 classes, top-5 is at least top-2
        acc_knn_top5 = acc_knn_top2 
        # A more robust way would be to check against knn.classes_[np.argsort(probas, axis=1)[:,-min(5, X_train.shape[0]):]]

    return acc_knn_top1, acc_knn_top2, acc_knn_top5

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load PyTorch Model
    model = DF_PyTorch(input_channels=1, emb_size=EMBEDDING_SIZE) # Make sure input_channels matches data
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}. Please check the path.")
        exit()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
    model.to(device)
    model.eval()
    print(f"Model {MODEL_PATH} loaded successfully.")

    # 2. Load Test Data (last 40 traces per site)
    site_trace_data_test = load_test_data(DATA_PATH, MAX_LEN)
    if not site_trace_data_test:
        print("No test data loaded. Exiting.")
        exit()
    print(f"Loaded test data for {len(site_trace_data_test)} sites.")

    # 3. Get Embeddings for all test traces
    # This can be memory intensive if all test data is large. 
    # Consider processing site by site if needed.
    all_site_embeddings_test = get_embeddings(model, device, site_trace_data_test)
    if not all_site_embeddings_test:
        print("Failed to generate any embeddings. Exiting.")
        exit()
    print(f"Generated embeddings for {len(all_site_embeddings_test)} sites.")

    # 4. Perform Evaluation Runs
    for sop in SIZE_OF_PROBLEM_LIST:
        print(f"\n--- Size of Problem (SOP): {sop} ---")
        for n_shot in N_SHOT_LIST:
            print(f"  --- N-Shot: {n_shot} ---")
            
            current_top1_accuracies = []
            current_top2_accuracies = []
            current_top5_accuracies = []

            for run_idx in range(NUM_EVAL_RUNS):
                # For each run, create new signature/test splits from the available sites for this SOP
                sig_vectors, sig_labels, tst_vectors, tst_labels = \
                    create_signature_and_test_sets(all_site_embeddings_test, n_shot, sop)
                
                if sig_vectors.shape[0] == 0 or tst_vectors.shape[0] == 0:
                    print(f"    Run {run_idx+1}/{NUM_EVAL_RUNS}: Skipped due to insufficient data for signature/test split.")
                    continue

                # Determine n_neighbors for kNN based on n_shot and TYPE_EXP
                n_neighbors_for_knn = n_shot if TYPE_EXP == "N-MEV" else 1 # If RAW, usually 1-NN per actual signature instance
                # Or, if TYPE_EXP is RAW and you have multiple raw signatures per class, n_shot could still be used.
                # For N-MEV, each class has 1 mean vector, so n_neighbors for kNN is effectively the number of mean vectors per class (which is 1 here)
                # The original kNN used n_shot as n_neighbors. Let's stick to that for consistency if TYPE_EXP is N-MEV.
                # If TYPE_EXP is 'RAW', and we have n_shot raw embeddings per class in X_train, then n_shot neighbors makes sense.
                
                # The number of neighbors for kNN should be based on how many training *samples* (signature vectors) there are.
                # If N-MEV, X_train has one sample per class. So kNN neighbors should be <= number of classes.
                # If RAW, X_train has n_shot samples per class. So kNN neighbors can be up to n_shot.
                # The original script uses n_shot for n_neighbors in kNN. Let's maintain that.
                actual_n_neighbors_knn = min(n_shot, sig_vectors.shape[0]) # Cannot have more neighbors than training samples
                if actual_n_neighbors_knn == 0:
                    print(f"    Run {run_idx+1}/{NUM_EVAL_RUNS}: Skipped, 0 neighbors for kNN.")
                    continue

                acc1, acc2, acc5 = knn_accuracy(sig_vectors, sig_labels, tst_vectors, tst_labels, actual_n_neighbors_knn)
                current_top1_accuracies.append(acc1)
                current_top2_accuracies.append(acc2)
                current_top5_accuracies.append(acc5)
                # print(f"    Run {run_idx+1}/{NUM_EVAL_RUNS}: Top-1: {acc1:.4f}, Top-2: {acc2:.4f}, Top-5: {acc5:.4f}")

            if current_top1_accuracies:
                print(f"    Avg Top-1 Accuracy: {np.mean(current_top1_accuracies):.4f} (Std: {np.std(current_top1_accuracies):.4f})")
                print(f"    Avg Top-2 Accuracy: {np.mean(current_top2_accuracies):.4f} (Std: {np.std(current_top2_accuracies):.4f})")
                print(f"    Avg Top-5 Accuracy: {np.mean(current_top5_accuracies):.4f} (Std: {np.std(current_top5_accuracies):.4f})") 
                print(f"    Raw Top-1: {str([float(f'{x:.4f}') for x in current_top1_accuracies]).strip('[]')}")
            else:
                print("    No successful evaluation runs for this configuration.")