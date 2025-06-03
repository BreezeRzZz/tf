import os
import random
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset # <--- 添加 TensorDataset 到这里
from DF_model_pytorch import DF_PyTorch # Import the PyTorch model
import torch.nn.functional as F

# Hyperparameters (same as before, or adjust as needed)
alpha = 0.1
batch_size_value = 128
emb_size = 64
number_epoch = 30
learning_rate = 0.001 # Keras SGD default lr is 0.01, but momentum and nesterov were used.
                      # Adam is a common PyTorch optimizer.

description = 'Triplet_Model_PyTorch'
Training_Data_PATH = '../../dataset/ts_mon.pkl' # Using the new data path

print(Training_Data_PATH)
print(f"with parameters, Alpha: {alpha}, Batch_size: {batch_size_value}, Embedded_size: {emb_size}, Epoch_num: {number_epoch}")
print(description)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading and Preprocessing (adapted from your previous Keras version) ---
with open(Training_Data_PATH, 'rb') as handle:
    dataset = pickle.load(handle)

site_ids = sorted(list(dataset.keys()))
num_classes = len(site_ids)
print(f"number of classes: {num_classes}")

name_to_classid = {site_id: i for i, site_id in enumerate(site_ids)}
classid_to_name = {i: site_id for i, site_id in enumerate(site_ids)}

all_traces_list = []
id_counter = 0
classid_to_ids = {i: [] for i in range(num_classes)}
id_to_classid = {}

for original_site_id, traces_for_site in dataset.items():
    traces_for_site = traces_for_site[:160]
    current_class_id = name_to_classid[original_site_id]
    for trace_data in traces_for_site:
        directions = [np.sign(x) if x != 0 else 1 for x in trace_data]
        # Ensure trace_data is a numpy array of fixed length (e.g., 5000)
        trace_arr = np.array(directions, dtype=np.float32) # Ensure float32 for PyTorch
        if len(trace_arr) > 5000:
            trace_arr = trace_arr[:5000]
        elif len(trace_arr) < 5000:
            trace_arr = np.pad(trace_arr, (0, 5000 - len(trace_arr)), 'constant', constant_values=0)
        all_traces_list.append(trace_arr)
        classid_to_ids[current_class_id].append(id_counter)
        id_to_classid[id_counter] = current_class_id
        id_counter += 1

all_traces_np = np.array(all_traces_list)
# For PyTorch Conv1D, input should be (N, C_in, L_in)
# Original Keras input was (N, L_in, C_in) after all_traces[:, :, np.newaxis]
# So, if all_traces_np is (N, 5000), we need to make it (N, 1, 5000)
if all_traces_np.ndim == 2:
    all_traces_np = all_traces_np[:, np.newaxis, :] # Shape: (N, 1, 5000)

print(f"Load traces with {all_traces_np.shape}")
print(f"Total size allocated on RAM : {str(all_traces_np.nbytes / 1e6)} MB")

# --- Triplet Generation Logic (similar to Keras version) ---
def build_pos_pairs_for_id(classid):
    traces = classid_to_ids[classid]
    pos_pairs = [(traces[i], traces[j]) for i in range(len(traces)) for j in range(i + 1, len(traces))]
    random.shuffle(pos_pairs)
    return pos_pairs

def build_positive_pairs(class_id_range):
    listX1 = []
    listX2 = []
    for class_id in class_id_range:
        pos = build_pos_pairs_for_id(class_id)
        for pair in pos:
            listX1.append(pair[0])
            listX2.append(pair[1])
    perm = np.random.permutation(len(listX1))
    return np.array(listX1)[perm], np.array(listX2)[perm]

Xa_train_indices, Xp_train_indices = build_positive_pairs(range(num_classes))
all_traces_train_idx = list(set(Xa_train_indices) | set(Xp_train_indices))

print(f"X_train Anchor indices: {Xa_train_indices.shape}")
print(f"X_train Positive indices: {Xp_train_indices.shape}")

# --- PyTorch Dataset and DataLoader for Triplet Mining ---
class TripletDataset(Dataset):
    def __init__(self, all_traces_data, anchor_indices, positive_indices, id_to_classid_map, classid_to_ids_map, num_classes, neg_traces_indices, model_for_similarity=None, alpha_val=0.1, device='cpu'):
        self.all_traces_data = torch.tensor(all_traces_data, dtype=torch.float32)
        self.anchor_indices = anchor_indices
        self.positive_indices = positive_indices
        self.id_to_classid_map = id_to_classid_map
        self.classid_to_ids_map = classid_to_ids_map
        self.num_classes = num_classes
        self.neg_traces_indices = neg_traces_indices # List of all valid indices for negative sampling
        self.model_for_similarity = model_for_similarity
        self.alpha_val = alpha_val
        self.device = device
        self.similarities = None
        self.similarity_matrix = None
        if self.model_for_similarity:
            self.update_similarities()

    def update_similarities(self):
        if self.model_for_similarity is None:
            self.similarities = None
            return
        print("Building similarities for hard negative mining...")
        self.model_for_similarity.eval() # Set model to evaluation mode
        with torch.no_grad():
            # Process in batches if all_traces_data is too large
            all_embs_list = []
            temp_loader = DataLoader(TensorDataset(self.all_traces_data), batch_size=batch_size_value, shuffle=False)
            for batch_data, in temp_loader:
                batch_data = batch_data.to(self.device)
                embs = self.model_for_similarity(batch_data)
                all_embs_list.append(embs.cpu())
            all_embs = torch.cat(all_embs_list, dim=0)
            all_embs = F.normalize(all_embs, p=2, dim=1)
            self.similarities = torch.matmul(all_embs, all_embs.T).numpy() # Store as numpy array
        self.model_for_similarity.train() # Set model back to train mode
        print("Similarities built.")

    def __len__(self):
        return len(self.anchor_indices)

    def __getitem__(self, idx):
        anc_idx = self.anchor_indices[idx]
        pos_idx = self.positive_indices[idx]

        anchor_trace = self.all_traces_data[anc_idx]
        positive_trace = self.all_traces_data[pos_idx]

        # Negative sampling (semi-hard or random)
        neg_idx = self._get_negative_sample(anc_idx, pos_idx)
        negative_trace = self.all_traces_data[neg_idx]
        
        return anchor_trace, positive_trace, negative_trace

    def _get_negative_sample(self, anc_idx, pos_idx):
        anchor_class = self.id_to_classid_map[anc_idx]
        positive_class = self.id_to_classid_map[pos_idx]
        
        if anchor_class != positive_class:
            # This case should ideally not happen if create_triplets is correct
            # but as a fallback, just pick a random different class sample
            print(f"Warning: Anchor class {anchor_class} and Positive class {positive_class} differ for anchor {anc_idx}, positive {pos_idx}.")
            while True:
                neg_cand_idx = random.choice(self.neg_traces_indices)
                if self.id_to_classid_map[neg_cand_idx] != anchor_class:
                    return neg_cand_idx

        neg_idx = -1

        # Try semi-hard negative mining using the precomputed similarity matrix
        if self.similarity_matrix is not None and anc_idx < self.similarity_matrix.shape[0]:
            try:
                # Get distances of anchor to all other samples
                anchor_similarities = self.similarity_matrix[anc_idx]
                dist_ap = anchor_similarities[pos_idx] # Similarity to positive

                # Find potential negatives: different class and dist_an > dist_ap
                potential_neg_indices = [
                    i for i, sim_an in enumerate(anchor_similarities) 
                    if self.id_to_classid_map[i] != anchor_class and sim_an > dist_ap # sim_an > dist_ap for cosine similarity (closer to 1 is more similar)
                ]
                
                if potential_neg_indices:
                    # Among these, find those that satisfy dist_an < dist_ap + margin (for cosine, this means sim_an > sim_ap - margin_adjusted_for_similarity)
                    # For L2 distance: dist_ap < dist_an < dist_ap + margin
                    # For cosine similarity (where 1 is most similar, -1 is least): sim_an < sim_ap AND sim_an > sim_ap - margin
                    # Let's assume lower similarity value means further, so we want: sim_ap > sim_an > sim_ap - margin
                    # (This part needs careful check based on how similarity_matrix is defined: distance or similarity score)
                    # Assuming similarity_matrix stores cosine similarity (higher is better)
                    # We want d(A,P) < d(A,N) < d(A,P) + margin. If using cosine similarity s, then s(A,P) > s(A,N) and s(A,N) > s(A,P) - margin.
                    
                    semi_hard_negatives = [
                        neg_cand 
                        for neg_cand in potential_neg_indices 
                        if anchor_similarities[neg_cand] < dist_ap and anchor_similarities[neg_cand] > dist_ap - self.margin 
                    ]
                    
                    if semi_hard_negatives:
                        neg_idx = random.choice(semi_hard_negatives)
                    else: # If no semi-hard, pick any valid negative from potential_neg_indices (hard or easy)
                        neg_idx = random.choice(potential_neg_indices)

            except (IndexError, ValueError) as e:
                # print(f"Semi-hard mining error for anchor {anc_idx}: {e}. Falling back to random.")
                pass # Fallback to random if any error

        # Fallback to random sampling if semi-hard mining failed or was not applicable
        if neg_idx == -1:
            # Create a pool of candidates that are not in the anchor_class
            # This is the crucial change to prevent infinite loop
            negative_candidate_pool = [idx for idx in self.neg_traces_indices if self.id_to_classid_map[idx] != anchor_class]
            
            if not negative_candidate_pool:
                # This should not happen if there's more than one class in the dataset.
                # If it does, it's a deeper issue with data or class mapping.
                print(f"Error: No negative samples found for anchor class {anchor_class}. Anchor idx: {anc_idx}")
                # As a last resort, just pick any sample that is not the anchor itself, though it might be from the same class.
                # This is not ideal but prevents a crash. Or raise an error.
                temp_pool = [idx for idx in self.neg_traces_indices if idx != anc_idx]
                if temp_pool: 
                    return random.choice(temp_pool)
                else: # Only one sample in the entire dataset? Highly unlikely.
                    raise ValueError("Cannot find any negative sample and dataset seems to have only one trace.")
            
            neg_idx = random.choice(negative_candidate_pool)
            
        return neg_idx

# --- Triplet Loss Function ---
class TripletLoss(nn.Module):
    def __init__(self, alpha, margin=1.0):
        super(TripletLoss, self).__init__()
        self.alpha = alpha
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dist_ap = F.pairwise_distance(anchor, positive, p=2)  # L2 distance
        dist_an = F.pairwise_distance(anchor, negative, p=2)  # L2 distance
        loss = torch.relu(dist_ap - dist_an + self.margin)
        return loss.mean(), dist_ap, dist_an # 返回 loss, dist_ap, dist_an

    def forward(self, anchor_emb, positive_emb, negative_emb):
        # Cosine similarity
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        pos_sim = cos_sim(anchor_emb, positive_emb)
        neg_sim = cos_sim(anchor_emb, negative_emb)
        
        losses = F.relu(neg_sim - pos_sim + self.alpha)
        return losses.mean()

# --- Model, Optimizer, Loss ---
model = DF_PyTorch(input_channels=1, emb_size=emb_size).to(device)

# Dynamically calculate flattened size and re-initialize FC layer
model.flattened_size = model._calculate_flattened_size(input_shape_l=all_traces_np.shape[2])
model.fc = nn.Linear(model.flattened_size, emb_size).to(device)
print(f"Model's calculated flattened size: {model.flattened_size}")

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True) # Alternative closer to Keras
triplet_loss_fn = TripletLoss(alpha=alpha)

# --- Training Loop ---
# At first epoch, no hard triplets (similarities=None)
train_dataset = TripletDataset(all_traces_np, Xa_train_indices, Xp_train_indices, 
                               id_to_classid, classid_to_ids, num_classes, 
                               all_traces_train_idx, model_for_similarity=None, alpha_val=alpha, device=device)
train_loader = DataLoader(train_dataset, batch_size=batch_size_value, shuffle=True, num_workers=0) # num_workers > 0 for parallel loading

for epoch in range(number_epoch):
    print(f"Epoch {epoch+1}/{number_epoch}")
    model.train() # Set model to training mode
    running_loss = 0.0
    
    # After first epoch, update similarities for semi-hard negative mining
    if epoch > 0:
        print("Updating similarities for semi-hard negative mining for new epoch...")
        train_dataset.model_for_similarity = model # Pass the current model state
        train_dataset.update_similarities()
        # Recreate DataLoader if dataset properties changed significantly, though here only similarities matrix is updated internally.
        # train_loader = DataLoader(train_dataset, batch_size=batch_size_value, shuffle=True, num_workers=0)

    for i, (anchor, positive, negative) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()

        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        loss = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0: # Log every 100 batches
            print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f}")
    
    # Save model (optional, e.g., every few epochs or at the end)
    # if (epoch + 1) % 5 == 0 or (epoch + 1) == number_epoch:
    save_path = f'trained_model/{description}_epoch{epoch+1}.pth'
    os.makedirs('trained_model', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

print("Training finished.")
# Save the final model
final_save_path = f'trained_model/{description}_final.pth'
_ = os.makedirs('trained_model', exist_ok=True) # Ensure directory exists
torch.save(model.state_dict(), final_save_path)
print(f"Final model saved to {final_save_path}")

# Training loop
for epoch in range(EPOCH_NUM):
    model.train()
    running_loss = 0.0
    running_corrects = 0 # 新增：用于累计正确的 triplet 数量
    running_total_triplets = 0 # 新增：用于累计总的 triplet 数量

    print(f'Epoch {epoch+1}/{EPOCH_NUM}')
    if epoch > 0: # Update similarities for semi-hard negative mining from the second epoch
        print("Updating similarities for semi-hard negative mining for new epoch...")
        train_dataset.update_similarities()

    for i, (anchor_idx, positive_idx, negative_idx) in enumerate(train_loader):
        anchor_emb_input = train_dataset.all_traces_data[anchor_idx].to(device)
        positive_emb_input = train_dataset.all_traces_data[positive_idx].to(device)
        negative_emb_input = train_dataset.all_traces_data[negative_idx].to(device)

        optimizer.zero_grad()

        anchor_emb = model(anchor_emb_input)
        positive_emb = model(positive_emb_input)
        negative_emb = model(negative_emb_input)
        
        # 修改这里以接收 dist_ap 和 dist_an
        loss, dist_ap_batch, dist_an_batch = loss_fn(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * anchor_emb.size(0) # loss.item() 是平均损失，乘以batch size
        
        # 计算准确率
        with torch.no_grad():
            correct_predictions = (dist_ap_batch < dist_an_batch).sum().item()
            running_corrects += correct_predictions
            running_total_triplets += anchor_emb.size(0)

        if (i + 1) % 100 == 0:
            # 在这里也打印批次准确率（可选）
            batch_acc = correct_predictions / anchor_emb.size(0) if anchor_emb.size(0) > 0 else 0
            print(f'  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Batch Acc: {batch_acc:.4f}')

    epoch_loss = running_loss / len(train_dataset) # 平均到每个triplet
    epoch_acc = running_corrects / running_total_triplets if running_total_triplets > 0 else 0 # 计算epoch准确率
    print(f'Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}')
    
    # Save model (optional, e.g., every few epochs or at the end)
    if (epoch + 1) % 5 == 0 or (epoch + 1) == number_epoch:
        save_path = f'trained_model/{description}_epoch{epoch+1}.pth'
        os.makedirs('trained_model', exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

print("Training finished.")
# Save the final model
final_save_path = f'trained_model/{description}_final.pth'
_ = os.makedirs('trained_model', exist_ok=True) # Ensure directory exists
torch.save(model.state_dict(), final_save_path)
print(f"Final model saved to {final_save_path}")
