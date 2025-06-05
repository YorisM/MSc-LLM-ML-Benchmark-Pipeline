
import os, sys, pickle, torch, gc, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

DATASET = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv('./challenges/FOURTOPS/data/X_train.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv',
                          dtype=np.int64 ).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv('./challenges/FOURTOPS/data/X_val.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv',
                          dtype=np.int64 ).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train),
            torch.from_numpy(Y_train),
            torch.from_numpy(X_val),
            torch.from_numpy(Y_val))

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        if isinstance(self.x, (tuple, list)):
            return (tuple(t[idx] for t in self.x), self.y[idx])
        else:
            return (self.x[idx], self.y[idx])      

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = PairDataset(X_train, Y_train)
    val_ds   = PairDataset(X_val , Y_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0))
                        
# ----------------  START OF LLM BLOCK  ----------------

python
Okay, I need to solve this machine learning problem where the goal is to maximize the AUC for classifying particle physics events as signal or background. Let me start by understanding the problem and the dataset.

First, the dataset consists of events with variable numbers of particles (objects), each represented by kinematic features. The data is zero-padded to a maximum of 18 objects per event. Each object has 5 features: obj identifier, E, pT, eta, phi. Additionally, there's missing transverse energy (E_T_miss and phi_Et_miss). The input tensor has 92 features, which breaks down as 2 for missing ET and 18 objects * 5 features each (but 18*5=90, so 2+90=92). 

The challenge is to process this variable-length data effectively. Since the objects are zero-padded, I need a way to handle the padding. Maybe using a mask to indicate which elements are real and which are padding. That way, the model can ignore the padded zeros during processing.

For preprocessing, I should normalize the features. But since the data is sparse and zero-padded, normalization should be done per-feature, ignoring the padded zeros. For example, compute mean and std for each feature (like E, pT, etc.) using only the non-zero entries. However, the first two features (E_T_miss and phi_Et_miss) are not part of the objects and should be normalized separately. Also, the obj identifier (integer) might be categorical. Maybe one-hot encode them? Or treat them as embeddings? But since they are part of the input tensor, perhaps normalization is needed here as well. Wait, the obj identifiers are integers representing particle types. Treating them as categorical variables might be better. So perhaps one-hot encoding each obj identifier. But since each object's first feature is obj_n (like obj_1, obj_2), which is an integer, maybe we can convert those into embeddings or one-hot vectors. However, the original code template's preprocessing step needs to transform the input tensor. Let's think: the input tensor is [N, 92], where each object's features start at positions 2-6, 7-11, etc. Each object has 5 features. So for each object, the first feature is the obj identifier (integer), followed by E, pT, eta, phi. 

Wait, the problem description says: "obj_1, E_1, p_T1, eta_1, phi_1". So for each object, the first entry is the obj identifier (like an integer representing the particle type), then E, pT, eta, phi. So each object's features are [obj_id, E, pT, eta, phi]. So for preprocessing, perhaps we can separate these into different features. For example, the obj_id could be one-hot encoded, and the other features normalized. But how to handle variable-length objects?

But the preprocessing step needs to transform the input tensor. The original input is a flat tensor of 92 elements per event. So for each event, the first two elements are E_T_miss and phi_Et_miss. Then, each object takes 5 elements. So for 18 objects, that's 18*5=90 elements, plus the first two gives 92.

So the preprocessing should reshape each event into a structure that represents the objects. For example, reshape the 92 elements into a structure with E_T_miss, phi_Et_miss, and a sequence of objects (each with 5 features). Then, process each object's features.

But how to handle variable-length sequences? Since the data is zero-padded, perhaps we can create a mask indicating which objects are real. For each object, if the obj_id is zero, then it's a padding. Wait, but the obj_id is an integer. So maybe if obj_id is zero, it's a padding. But according to the problem statement, the data is zero-padded. So for example, if an event has 5 objects, then the first 5 objects' entries are filled, and the remaining 13 are zeros. So for each object in the event, if all features (obj_id, E, pT, eta, phi) are zero, then it's padding. But maybe the obj_id is non-zero even for padding? Wait, the problem says "zero-padded to the largest number of objects". So perhaps the padding is zeros in all features. So for each object, if the obj_id is zero, then it's a padding. So we can create a mask where obj_id != 0.

So preprocessing steps:

1. Reshape each event into (E_T_miss, phi_Et_miss, objects), where objects is a tensor of shape (18,5). Each object has 5 features: obj_id, E, pT, eta, phi.

2. Separate E_T_miss and phi_Et_miss, normalize them.

3. For each object, extract obj_id, E, pT, eta, phi. Normalize E, pT, eta, phi per feature, ignoring the padded zeros. For obj_id, perhaps one-hot encode or use embeddings.

But how to handle normalization. For example, E is in MeV. So normalizing E, pT (which are in MeV) by their mean and std. But since the data is zero-padded, when computing mean and std, we should exclude the padded zeros.

So during preprocessing, in the fit method, compute the mean and std for each feature (E, pT, eta, phi) across all objects, but only for those objects where obj_id is non-zero. Similarly, compute mean and std for E_T_miss and phi_Et_miss (but these are always present, so just compute over all events).

Wait, E_T_miss and phi_Et_miss are present for every event, so their normalization can be done straightforwardly. For the object features, each event has up to 18 objects, but many are zero-padded. So for each object's E, pT, etc., we need to compute mean and std over all non-padded entries.

So the preprocessor needs to:

- Compute mean and std for E_T_miss and phi_Et_miss.

- For each object feature (E, pT, eta, phi), compute mean and std, considering only non-padded objects (i.e., where obj_id != 0).

But how to handle obj_id? Since obj_id is an integer representing the particle type, perhaps we can one-hot encode them. For example, if there are 10 possible obj_ids, then each obj_id is converted into a 10-dimensional one-hot vector. But we need to know the possible obj_ids in the dataset. Alternatively, treat obj_id as a categorical feature and use an embedding layer. However, embedding layers are part of the model, not preprocessing. So during preprocessing, perhaps we can one-hot encode the obj_ids. But how many possible obj_ids are there? The problem statement doesn't specify. So perhaps during preprocessing, collect all unique obj_ids present in the training data (excluding padding) and create a one-hot encoding based on that. But that's stateful and needs to be stored in the preprocessor.

Alternatively, during preprocessing, extract the obj_ids, collect all possible values, create a mapping to indices, then one-hot encode. But this requires the preprocessor to remember the possible obj_ids. So in the fit method, collect all obj_ids from the training data (excluding zeros) and determine the unique values. Then, during transform, map each obj_id to a one-hot vector. However, this could be memory-intensive if there are many obj_ids. Alternatively, use an embedding layer in the model, which would handle the obj_ids as categorical inputs. But embedding layers require the input to be integer indices. So perhaps during preprocessing, map each obj_id to an integer index (starting from 1, with 0 reserved for padding), and then the model can use an embedding layer. But how to handle obj_ids that are not seen during training? Since the data is simulated, perhaps all possible obj_ids are present in the training set.

So steps for preprocessing:

1. Reshape the input tensor (N, 92) into (N, 2 + 18*5) → but wait, 2 + 18*5 is 92. So each event is split into E_T_miss (1), phi_Et_miss (1), and 18 objects each with 5 features. So reshape each event into (2 + 18*5) → but that's already the input. Wait, the input is already given as a tensor of shape [N,92]. So each event is a vector of 92 elements. The first two elements are E_T_miss and phi_Et_miss. Then, each object is represented by 5 elements: obj_id, E, pT, eta, phi. There are up to 18 objects, hence 18*5=90 elements. So total 2+90=92.

So for preprocessing, each event can be reshaped into:

- E_T_miss (1)
- phi_Et_miss (1)
- objects (18, 5)

So the preprocessor can separate these parts.

Then, for each object, check if it's a padding (obj_id == 0). For non-padding objects, collect their features for normalization.

So during the fit phase:

- Compute mean and std for E_T_miss and phi_Et_miss over all events.

- For each of the 18 objects, collect E, pT, eta, phi values where obj_id != 0. Then compute mean and std for each of these features across all non-padded objects.

Additionally, for obj_id, collect all unique values (excluding zero) and create a mapping to indices for embedding or one-hot encoding.

But handling obj_ids in preprocessing is tricky. One approach is to one-hot encode obj_id during preprocessing, but that would increase the feature dimension. For example, if there are 10 unique obj_ids, each obj_id would be converted into a 10-dimensional vector. Then, each object's features would be 10 (one-hot) + E + pT + eta + phi = 14 features. But this would increase the input size significantly, especially if there are many obj_ids. Alternatively, use an embedding layer in the model, which would map each obj_id to a dense vector. This would require the model to handle the obj_id as an integer index.

So during preprocessing:

- For each object, extract obj_id. Replace obj_id with an integer index (if using embedding). For example, during fit, collect all unique obj_ids (excluding zero) and assign each a unique index (like 1, 2, ..., K). Then, during transform, replace each obj_id with its index (or zero for padding). Then, in the model, use an embedding layer with input_dim K+1 (since indices start at 0) and output_dim embedding_size.

But how to handle obj_ids in the validation set that are not present in the training set? Since the data is simulated, perhaps all possible obj_ids are present in the training set. So during fit, collect all unique obj_ids from the training data (excluding zero), assign each a unique index. Then, during transform, any obj_id not in the training set would be mapped to zero (padding) or a default index. But this may not be ideal. Alternatively, during preprocessing, map all obj_ids not seen during training to a special 'unknown' index. But this requires handling during fit.

Alternatively, during preprocessing, treat obj_id as a categorical feature and use one-hot encoding. For example, collect all unique obj_ids in the training data (excluding zero), create a one-hot vector for each. Then, during transform, if an obj_id is not present, treat it as zero (but this would be a problem). So perhaps during preprocessing, the obj_id is converted into a one-hot vector based on the training set's unique obj_ids. Any obj_id not in the training set would be treated as zero (padding), but since the data is simulated, this might not happen.

But this approach would require storing the list of unique obj_ids in the preprocessor. Let's proceed with this.

So steps for preprocessing:

In the fit method:

1. Extract E_T_miss and phi_Et_miss from all training events, compute their mean and std.

2. For each object in each training event:

   a. Check if obj_id is non-zero (i.e., not padding).

   b. Collect E, pT, eta, phi values for these objects.

   c. Compute mean and std for each of these four features.

3. Collect all unique obj_ids (excluding zero) from the training data. Create a mapping from obj_id to a one-hot index. For example, if the unique obj_ids are [1,3,5], then map 1→0, 3→1, 5→2. Then, during transform, each obj_id is replaced with this index. Then, in the model, an embedding layer can be used with input_dim equal to the number of unique obj_ids + 1 (for padding).

Wait, but embedding layers require the input to be indices. So during preprocessing, replace each obj_id with an index (starting from 1 for the first unique obj_id, etc.), and zero for padding. Then, the embedding layer can map these indices to dense vectors.

So during preprocessing:

- For each object's obj_id, replace it with an index based on the unique obj_ids found in training. For example, if the unique obj_ids are [5, 10, 15], then during fit, create a dictionary {5:1, 10:2, 15:3}, and during transform, replace obj_id 5 with 1, 10 with 2, etc. Any obj_id not in the training set (unlikely) would be treated as padding (index 0). But since the data is simulated, perhaps all obj_ids in validation are present in training.

So the preprocessor needs to store a list of unique obj_ids encountered during fit. Then, during transform, each obj_id is mapped to its index (starting from 1), and zero if not found (but this shouldn't happen).

But how to handle this in code:

In the preprocessor's fit method:

Loop through all training events, extract all obj_ids (from the objects) where obj_id !=0. Collect all unique obj_ids, sort them, assign each a unique index starting from 1. Store this mapping as a dictionary in the preprocessor.

Then, during transform:

For each object in each event, check if obj_id is non-zero. If so, look up its index in the mapping. If not found, treat as zero (padding). Then, replace the obj_id with this index. The other features (E, pT, eta, phi) are normalized using the mean and std computed during fit.

So the transformed data would have:

- E_T_miss and phi_Et_miss normalized.

- Each object's obj_id replaced with an index (0 for padding, 1,2,...K for known obj_ids).

- Each object's E, pT, eta, phi normalized.

But how to structure the transformed data. The original input is a flat tensor of 92 elements. After preprocessing, perhaps we reshape it into a structure that can be fed into a model that handles sequences (like a transformer or RNN). For example, each event becomes:

- A tensor of shape [2] for E_T_miss and phi_Et_miss (normalized).

- A tensor of shape [18, 5] for the objects, where each object's features are [indexed_obj_id, normalized E, normalized pT, normalized eta, normalized phi].

Wait, but after replacing obj_id with an index, each object's first feature is the index (integer), and the other four are normalized. So each object has 5 features: index (int), E (float), pT (float), eta (float), phi (float). But the index is an integer, which the model can process via an embedding layer.

Alternatively, during preprocessing, separate the obj_id from the other features. For example, each object is split into obj_id (index) and the four kinematic features. Then, the model can process the obj_id through an embedding layer and concatenate with the normalized kinematic features.

But how to structure this in the transformed tensor. Let's think:

Original input per event: [E_T_miss, phi_Et_miss, obj_1, E_1, pT_1, eta_1, phi_1, obj_2, E_2, ...]

After preprocessing:

E_T_miss and phi_Et_miss are normalized.

Each obj_i is replaced with an index (0 for padding, 1-K for known obj_ids).

Each E_i, pT_i, eta_i, phi_i are normalized.

So the transformed data per event would be:

A flat tensor of 2 (E_T_miss, phi) + 18 objects * 5 features each (indexed obj_id, E, pT, eta, phi). But the indexed obj_id is an integer, and the others are normalized floats.

But how to handle this in the model. The model would need to process the obj_id as an embedding and combine it with the other features.

So perhaps the transformed data is split into two parts: the global features (E_T_miss and phi) and the object sequences. Then, for each object, the model uses an embedding layer on the obj_id index and concatenates it with the normalized kinematic features.

But integrating this into the preprocessing step requires restructuring the data. Alternatively, during preprocessing, separate the obj_id indices and the kinematic features into different tensors. For example, the transformed data could be a tuple of (global_features, obj_indices, obj_features), but the code template allows the preprocessor's transform to return either a single tensor or a tuple of tensors (like (X_seq, mask)).

Wait, the code template says that the preprocessor's transform can return X_new (N, features) or (X_seq, mask) (N, L, F), (N, L). So perhaps the preprocessor can reshape the data into a sequence of objects, each with their features, along with a mask indicating which objects are real.

So here's a plan:

1. Preprocess E_T_miss and phi_Et_miss: normalize them using their mean and std.

2. For each object in the event:

   a. Check if obj_id is non-zero. If zero, mask it out.

   b. Replace obj_id with the index from the preprocessor's mapping.

   c. Normalize E, pT, eta, phi using the mean and std computed during fit (only from non-padded objects).

3. Reshape the objects into a sequence of shape (18, 5), where each object has [indexed_obj_id, E_norm, pT_norm, eta_norm, phi_norm].

4. Create a mask where each object is 1 if obj_id was non-zero (original obj_id, not the indexed one), else 0.

Wait, but the mask is based on the original obj_id being non-zero. Because after preprocessing, the indexed_obj_id could be zero (if the original obj_id was zero, i.e., padding). So the mask would be 1 for objects that are not padding.

So during preprocessing:

For each event:

- Extract E_T_miss and phi_Et_miss, normalize.

- For each of the 18 objects:

   - Extract obj_id, E, pT, eta, phi.

   - If obj_id ==0: mask is 0, and all features are zero (but after normalization, they may not be zero). Wait, but during preprocessing, padded objects have all features zero. So during fit, when computing mean and std for E, pT, etc., we exclude objects where obj_id ==0.

But during transform, when we process an object with obj_id ==0 (padding), we still need to normalize E, pT, etc. But since the original features are zero, after normalization, they would be (0 - mean)/std, which may not be zero. That's a problem because padding should be treated as such. So perhaps during preprocessing, after normalization, we set the features of padded objects to zero. But how?

Alternatively, during preprocessing, for padded objects (obj_id ==0), set their E, pT, eta, phi to zero after normalization. But how to do that.

Wait, during preprocessing:

For each object in the event:

if original obj_id ==0 → it's padding. So during transform:

- Set indexed_obj_id to 0.

- Set E, pT, eta, phi to zero (even after normalization).

But wait, during normalization, E, pT, etc. are transformed as (x - mean)/std. For padded objects, the original E, pT, etc. are zero. So after normalization, they would be (-mean)/std. Which is not zero. So to avoid this, during preprocessing, after normalization, for padded objects, set their E, pT, eta, phi to zero.

But how to know which objects are padded? Based on the original obj_id. So during preprocessing:

For each object:

original_obj_id = X[..., 0] (the first feature of the object's 5 features).

if original_obj_id ==0 → it's a padding. So:

- indexed_obj_id =0

- E, pT, eta, phi are set to zero.

Else:

- indexed_obj_id = mapping[original_obj_id]

- E, pT, eta, phi are normalized.

But this way, padded objects have all features (indexed_obj_id, E, pT, etc.) set to zero, except for the original obj_id which was zero.

Wait, but during preprocessing, the original obj_id is part of the input. So for each object's features in the input tensor, the first element is the original obj_id. So during preprocessing:

For each object in the event (each group of 5 features):

original_obj_id = X[i, 2 + 5*j] where j is the object index (0-17).

Wait, the input tensor is structured as:

Indices 0-1: E_T_miss, phi_Et_miss.

Indices 2-6: object 1 (obj_1, E_1, pT_1, eta_1, phi_1).

Indices 7-11: object 2.

So for each object j (0-based), the starting index is 2 + 5*j.

So for object j, the features are:

obj_id = X[i, 2 +5j]

E = X[i, 2 +5j +1]

pT = X[i, 2 +5j +2]

eta = X[i, 2 +5j +3]

phi = X[i, 2 +5j +4]

So during preprocessing, for each object j in 0..17:

Check if obj_id is zero. If yes → padding. So during transform:

- Set indexed_obj_id to 0.

- Set E, pT, eta, phi to zero.

Else:

- Replace obj_id with its index from the mapping.

- Normalize E, pT, eta, phi using the precomputed mean and std.

But after normalization, E, pT, etc. are transformed. However, for padded objects, their original E, pT, etc. are zero, but after normalization, they would be (0 - mean)/std. Which is incorrect. So during preprocessing, after normalization, for padded objects, set E, pT, eta, phi to zero.

So the steps for each object during transform:

if original_obj_id ==0:

   indexed_obj_id =0

   E =0

   pT=0

   eta=0

   phi=0

else:

   indexed_obj_id = mapping.get(original_obj_id, 0)  # 0 if not found (unlikely)

   E = (E - E_mean) / E_std

   pT = (pT - pT_mean) / pT_std

   eta = (eta - eta_mean) / eta_std

   phi = (phi - phi_mean) / phi_std

But wait, phi is an angle. Normalizing angles can be tricky. For example, phi is between -pi and pi. Normalizing phi by subtracting mean and dividing by std may not be appropriate. Perhaps better to scale it to [-1,1] by dividing by pi. Similarly for eta, which can be any real number but typically in a certain range. But the problem statement says the data is not preprocessed, so we need to handle normalization ourselves.

Alternatively, compute mean and std for phi and eta as well, but considering their distributions. However, for angles, using trigonometric features (like sin and cos) might be better. For example, instead of using phi directly, compute sin(phi) and cos(phi). Similarly for phi_Et_miss. This could help the model handle the periodicity of angles.

So perhaps during preprocessing, for phi_Et_miss and each object's phi, replace them with sin(phi) and cos(phi). Similarly for phi_Et_miss.

This would convert each phi into two features, which captures the periodic nature. This is a common practice in machine learning when dealing with angles.

So modifying the preprocessing steps:

For phi_Et_miss and each object's phi:

Compute sin(phi) and cos(phi).

Replace the original phi with these two values.

This increases the number of features but captures the angle's periodicity.

So for each event:

Original features:

E_T_miss, phi_Et_miss, obj_1, E_1, pT_1, eta_1, phi_1, obj_2, E_2, pT_2, eta_2, phi_2, ...

After preprocessing:

E_T_miss (normalized), sin(phi_Et_miss), cos(phi_Et_miss), and for each object:

indexed_obj_id, E (normalized), pT (normalized), eta (normalized), sin(phi), cos(phi).

Wait, but phi_Et_miss is a single feature. So replacing phi_Et_miss with sin and cos would add one more feature. So the global features would be E_T_miss (normalized), sin_phi_Et_miss, cos_phi_Et_miss → 3 features.

Each object's phi is replaced with sin and cos, adding one more feature per object. Originally, each object had 5 features (obj_id, E, pT, eta, phi). After replacing phi with sin and cos, each object has 6 features: obj_id, E, pT, eta, sin_phi, cos_phi.

But this increases the number of features per object from 5 to 6. So the total number of features per event would be:

3 (global) + 18 objects *6 features → 3 + 108 = 111 features. But the original input is 92 features. So this would require restructuring the data.

Alternatively, during preprocessing, restructure the data into a sequence of objects with their features, plus the global features.

But how to handle this in the code. Let's outline the steps:

1. Preprocess E_T_miss: normalize using mean and std.

2. Preprocess phi_Et_miss: compute sin and cos.

3. For each object:

   a. If obj_id is zero → padding: set indexed_obj_id to 0, E, pT, eta, sin_phi, cos_phi to zero.

   b. Else: indexed_obj_id from mapping, normalize E, pT, eta, compute sin and cos of phi.

So each object now has 6 features: indexed_obj_id, E_norm, pT_norm, eta_norm, sin_phi, cos_phi.

The global features are E_T_miss_norm, sin_phi_Et_miss, cos_phi_Et_miss → 3 features.

So the transformed data per event would be:

global_features (3) + objects (18,6) → total 3 + 18*6 = 111 features. But the original input is 92 features. So this requires restructuring the data into a different format.

But the code template allows the preprocessor to return either a single tensor (N, features) or a tuple of (X_seq, mask) where X_seq is (N, L, F) and mask is (N, L). So perhaps the preprocessor can return a tuple where X_seq is the sequence of objects (each with 6 features) and mask indicates which objects are real. Additionally, the global features can be concatenated with the sequence features or processed separately.

Alternatively, concatenate the global features with each object's features. For example, each object's features include the global features. But this would duplicate the global features for each object, which might not be efficient.

Another approach is to process the global features separately and combine them with the aggregated object features. For example, use a neural network that processes the sequence of objects (using an RNN or transformer), aggregates them into a vector, then concatenates with the processed global features, and then makes a prediction.

So the model architecture could have two branches: one for processing the sequence of objects and another for processing the global features. The outputs of these branches are concatenated and fed into a classifier.

But how to structure the data for this. The preprocessor would need to split the data into global features and object sequences.

So during preprocessing:

Each event is transformed into:

- global_features: tensor of shape [3] (E_T_miss_norm, sin_phi_Et_miss, cos_phi_Et_miss)

- objects: tensor of shape [18,6] (indexed_obj_id, E_norm, pT_norm, eta_norm, sin_phi, cos_phi)

- mask: tensor of shape [18] indicating which objects are real (1) or padding (0)

But how to return this from the preprocessor's transform method. The code template allows returning a tuple (X_seq, mask), which would be (N, L, F) and (N, L). But in this case, X_seq would be the objects (18,6), and mask is (18). But the global features are separate. So perhaps the preprocessor needs to return a tuple of (global_features, objects, mask), but the code template's transform method can return either a single tensor or a tuple of two tensors. So this complicates things.

Alternatively, concatenate the global features to each object's features. For example, each object's features include the global features. So each object's features would be [E_T_miss_norm, sin_phi_Et_miss, cos_phi_Et_miss, indexed_obj_id, E_norm, pT_norm, eta_norm, sin_phi, cos_phi]. This way, each object has 3+6=9 features, and the sequence is 18 objects. But this duplicates the global features for each object, which may not be efficient but allows processing in a single sequence.

Alternatively, process the global features separately and combine them with the aggregated object features. For example:

- Process the sequence of objects using a transformer or RNN to get a context vector.

- Process the global features using a dense network.

- Concatenate the context vector and global features vector, then pass through a classifier.

This approach requires the model to handle two separate inputs: the sequence of objects (with mask) and the global features.

But according to the code template, the model's forward function must match the data format returned by the preprocessor. So if the preprocessor returns a tuple (X_seq, mask) and global_features, then the model's forward would need to accept multiple inputs. But the code template's train_model function expects the data loader to yield either (data, label) or ((data, mask), label). So perhaps the preprocessor can return a tuple where data is a tuple of (global_features, objects, mask), and the model's forward function accepts this.

But this complicates the code structure. Alternatively, the preprocessor can return a single tensor that combines global features and object sequences in a way that the model can process. For example, each event is represented as a sequence where the first element is a "global" token containing the global features, followed by the object tokens. This is similar to how BERT uses [CLS] tokens.

So the transformed data would be a sequence of 1 (global) + 18 (objects) = 19 elements. Each element has its own features. The global token's features are the global features (E_T_miss_norm, sin_phi_Et_miss, cos_phi_Et_miss), and each object token has its 6 features. The mask would be 1 for the global token and the real objects, 0 for padded objects. But this approach may not be necessary. Alternatively, the global features can be processed separately and combined with the aggregated object features.

Given the complexity, perhaps the best approach is to process the objects as a sequence with mask, and concatenate the global features to each object's features. This way, each object's features include both its own attributes and the global event attributes. This allows the model to learn interactions between the objects and the global features.

So during preprocessing:

For each event:

- Compute global_features: [E_T_miss_norm, sin_phi_Et_miss, cos_phi_Et_miss]

- For each object j in 0..17:

   a. If original_obj_id ==0 → padding: set all features to zero, mask=0.

   b. Else: indexed_obj_id, normalized E, pT, eta, sin_phi, cos_phi.

   c. Concatenate global_features to the object's features → each object has 3 +6 =9 features.

So the transformed data per event is a sequence of 18 objects, each with 9 features (3 global + 6 object-specific), and a mask indicating which objects are real.

But this increases the sequence length to 18 and each object's features to 9. The mask is of shape (18).

But how to structure this in the code. The preprocessor's transform method would need to return a tuple of (X_seq, mask), where X_seq is (N, 18, 9) and mask is (N, 18).

But how to compute this:

In code:

def transform(self, X):
    # X is (N, 92)
    N = X.shape[0]
    # Reshape into global features and objects
    global_features = X[:, :2]  # E_T_miss, phi_Et_miss
    objects = X[:, 2:].view(N, 18, 5)  # (N, 18, 5)

    # Normalize global features
    # E_T_miss: normalize using self.et_miss_mean and self.et_miss_std
    # phi_Et_miss: compute sin and cos
    et_miss = (global_features[:, 0] - self.et_miss_mean) / self.et_miss_std
    phi_et = global_features[:, 1]
    sin_phi_et = torch.sin(phi_et)
    cos_phi_et = torch.cos(phi_et)
    global_features_processed = torch.stack([et_miss, sin_phi_et, cos_phi_et], dim=1)  # (N, 3)

    # Process each object
    obj_indices = objects[..., 0]  # (N, 18)
    obj_E = objects[..., 1]
    obj_pT = objects[..., 2]
    obj_eta = objects[..., 3]
    obj_phi = objects[..., 4]

    # Create mask: 1 where obj_id !=0, else 0
    mask = (obj_indices != 0).float()  # (N, 18)

    # Replace obj_id with indices from mapping
    # For each obj_id in obj_indices, replace with self.obj_id_mapping.get(id, 0)
    # But how to vectorize this in PyTorch?
    # This is tricky. Maybe loop through each element and replace.
    # Alternatively, precompute a lookup tensor.
    # But for now, let's assume that self.obj_id_mapping is a dictionary.
    # Convert obj_indices to a numpy array, apply mapping, then back to tensor.
    # But this requires detaching and moving to CPU.
    obj_indices_np = obj_indices.cpu().numpy()
    mapped_indices = np.zeros_like(obj_indices_np)
    for i in range(N):
        for j in range(18):
            original_id = obj_indices_np[i, j]
            if original_id !=0:
                mapped_indices[i,j] = self.obj_id_mapping.get(original_id, 0)  # 0 if not found
    mapped_indices = torch.from_numpy(mapped_indices).to(X.device)

    # Normalize E, pT, eta
    obj_E_norm = (obj_E - self.E_mean) / self.E_std
    obj_pT_norm = (obj_pT - self.pT_mean) / self.pT_std
    obj_eta_norm = (obj_eta - self.eta_mean) / self.eta_std

    # Compute sin and cos of phi
    obj_sin_phi = torch.sin(obj_phi)
    obj_cos_phi = torch.cos(obj_phi)

    # Combine object features: [mapped_indices, E_norm, pT_norm, eta_norm, sin_phi, cos_phi]
    # But mapped_indices is (N,18), others are (N,18)
    # So stack along the last dimension
    obj_features = torch.stack([
        mapped_indices,
        obj_E_norm,
        obj_pT_norm,
        obj_eta_norm,
        obj_sin_phi,
        obj_cos_phi
    ], dim=2)  # (N,18,6)

    # Now, concatenate global features to each object's features
    # global_features_processed is (N,3)
    # For each object in the sequence, repeat the global features
    # So expand global_features_processed to (N, 18, 3) and concatenate with obj_features (N,18,6)
    global_features_expanded = global_features_processed.unsqueeze(1).expand(-1, 18, -1)  # (N,18,3)
    combined_features = torch.cat([global_features_expanded, obj_features], dim=2)  # (N,18,9)

    # For padded objects, set features to zero
    # mask is (N,18), 1 for real, 0 for padded
    # Multiply each object's features by mask[:, :, None] to zero out padded objects
    combined_features = combined_features * mask[:, :, None]

    return (combined_features, mask)

But this approach has a few issues:

1. Handling the obj_id mapping requires converting to numpy, which might be slow and not GPU-compatible. However, during preprocessing, it's done once, so maybe acceptable.

2. The mask is applied after combining features, which zeros out the padded objects' features, including the global features. But the global features are the same for all objects in the event. However, for padded objects, their features are zeroed out, which is correct.

But wait, the global features are part of each object's features. For padded objects, their entire feature vector (including global features) is zeroed out. But the global features are event-level, not object-level. So this approach may not be correct. For example, the global features should be present for all objects, including padded ones. But padded objects are not real, so their features should be zero. So perhaps this is correct.

Alternatively, the global features should be included in each object's features, but padded objects' features (including global) are zero. This way, the model can still process the global features through the non-padded objects.

But this might not be the best approach. Perhaps the global features should be processed separately and not concatenated to each object's features. Let's reconsider.

Alternative approach:

Process the global features separately from the object sequence. The model can have two branches: one for the global features (processed through a dense network) and one for the object sequence (processed through a transformer or LSTM). The outputs of both branches are concatenated and fed into the classifier.

In this case, the preprocessor's transform method would return a tuple of ( (global_features, objects), mask ), but the code template allows returning a tuple of two tensors. However, the code template's train_model function expects the data loader to yield either (data, label) or ((data, mask), label). So if the preprocessor returns a tuple of (global, objects, mask), the data loader would yield ((global, objects, mask), label), which the model's forward would need to handle.

But integrating this into the code requires modifying the model to accept multiple inputs. For example:

class MyModel(nn.Module):
    def __init__(self, input_shape, use_mask=False):
        super().__init__()
        # input_shape would be a tuple, but in this case, the input is a tuple of (global_features, objects)
        # So need to define separate branches for each part.

        # Global features: 3 features
        self.global_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Object sequence: each object has 6 features (indexed_obj_id, E, pT, eta, sin_phi, cos_phi)
        self.obj_embedding = nn.Embedding(num_embeddings=self.num_obj_ids, embedding_dim=8)
        self.obj_net = nn.Linear(8 +5, 64)  # embedding + E, pT, eta, sin_phi, cos_phi

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8),
            num_layers=2
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 +64, 128),  # global + sequence
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask=None):
        global_features, objects = x
        # global_features: (N,3)
        # objects: (N,18,6) → [indexed_obj_id, E, pT, eta, sin_phi, cos_phi]

        # Process global features
        global_out = self.global_net(global_features)  # (N,64)

        # Process objects
        obj_ids = objects[..., 0].long()  # (N,18)
        obj_kinematics = objects[..., 1:]  # (N,18,5)

        # Embed obj_ids
        obj_emb = self.obj_embedding(obj_ids)  # (N,18,8)
        # Concatenate with kinematics
        obj_features = torch.cat([obj_emb, obj_kinematics], dim=2)  # (N,18,8+5=13)
        # Project to 64
        obj_features = self.obj_net(obj_features)  # (N,18,64)

        # Apply transformer
        # Mask: (N,18) where 1 is real, 0 is padding. For transformer, need to convert to attention mask.
        # Transformer expects src_key_padding_mask where True is padding (mask==0)
        src_key_padding_mask = (mask == 0)
        obj_features = obj_features.permute(1,0,2)  # (18, N, 64)
        transformer_out = self.transformer(obj_features, src_key_padding_mask=src_key_padding_mask)
        transformer_out = transformer_out.permute(1,0,2)  # (N,18,64)

        # Aggregate sequence: take mean over non-padded objects
        # Sum over sequence, divide by number of real objects
        seq_mask = mask.unsqueeze(-1)  # (N,18,1)
        sum_out = (transformer_out * seq_mask).sum(dim=1)  # (N,64)
        count = seq_mask.sum(dim=1)  # (N,1)
        count = torch.clamp(count, min=1e-5)  # avoid division by zero
        seq_out = sum_out / count  # (N,64)

        # Concatenate global and sequence features
        combined = torch.cat([global_out, seq_out], dim=1)  # (N, 128)
        logit = self.classifier(combined).squeeze(1)  # (N,)
        return logit

But this requires the model to handle two separate inputs: global_features and objects, along with the mask. However, according to the code template, the preprocessor's transform method can return either a single tensor or a tuple of two tensors (data and mask). So if the preprocessor returns a tuple of (global_features, objects, mask), the data loader would yield ((global_features, objects, mask), labels), which the model's forward method would need to handle. But the code template's train_model function expects the model's forward to accept either data or (data, mask). So this complicates things.

Alternatively, the preprocessor can return a tuple where data is a tuple of (global_features, objects), and mask is a separate tensor. Then, the model's forward would receive ( (global, objects), mask ), and process accordingly.

But integrating this into the code requires careful handling. For example, in the training loop, when getting a batch, the data would be a tuple of (global, objects), and mask. But PyTorch's DataLoader expects each batch to be a tuple of (data, label), where data can be a tuple. So this is possible.

But given the complexity, perhaps a better approach is to process the objects as a sequence with mask and include the global features as part of each object's features. This way, the model can process the entire event as a sequence, with each object's features including the global features. This simplifies the data structure returned by the preprocessor to a tuple of (X_seq, mask), where X_seq is (N, 18, 9) (3 global + 6 object features) and mask is (N, 18).

So the preprocessor's transform method would return (X_seq, mask), and the model's forward would process X_seq and mask.

The model can then use a transformer to process the sequence, using the mask to ignore padded objects. The transformer's output can be aggregated (e.g., mean pooling) and passed through a classifier.

This approach avoids handling multiple inputs and keeps the data structure simple.

So the model architecture would be:

1. Embed the obj_id indices using an embedding layer.

2. Concatenate the embedded obj_id with the normalized kinematic features and the global features.

3. Process the sequence with a transformer encoder.

4. Aggregate the sequence (e.g., mean pooling over non-padded objects).

5. Pass through a classifier.

But wait, the global features are already concatenated to each object's features. So each object's features include the global features. This allows the model to learn interactions between the objects and the global features.

So the model's forward would look like:

def forward(self, x, mask=None):
    # x: (N, 18, 9) → 3 global + 6 object features (including obj_id index)
    # mask: (N,18)

    # Split the features into obj_id index and others
    obj_indices = x[..., 0].long()  # (N,18)
    other_features = x[..., 1:]  # (N,18, 8)

    # Embed obj_indices
    obj_emb = self.embedding(obj_indices)  # (N,18, embedding_dim)

    # Concatenate with other features
    combined = torch.cat([obj_emb, other_features], dim=-1)  # (N,18, embedding_dim +8)

    # Project to transformer dimension
    combined = self.input_proj(combined)  # (N,18, d_model)

    # Transformer encoder
    # mask: (N,18) → src_key_padding_mask is True for padding
    src_key_padding_mask = (mask == 0)
    # Transformer expects seq_len first
    combined = combined.permute(1,0,2)  # (18, N, d_model)
    transformer_out = self.transformer(combined, src_key_padding_mask=src_key_padding_mask)
    transformer_out = transformer_out.permute(1,0,2)  # (N,18, d_model)

    # Aggregate: mean over non-padded objects
    seq_mask = mask.unsqueeze(-1)  # (N,18,1)
    sum_out = (transformer_out * seq_mask).sum(dim=1)  # (N, d_model)
    count = seq_mask.sum(dim=1)  # (N,1)
    count = torch.clamp(count, min=1e-5)
    mean_out = sum_out / count  # (N, d_model)

    # Classifier
    logit = self.classifier(mean_out).squeeze(1)  # (N,)
    return logit

This way, the model processes the entire sequence, including the global features concatenated to each object's features. The transformer can learn to attend to relevant objects and their relation to the global event features.

Now, implementing the preprocessor:

The preprocessor needs to:

- Compute mean and std for E_T_miss.

- Compute mean and std for E, pT, eta across non-padded objects.

- Collect unique obj_ids and create a mapping to indices.

- Replace obj_ids with indices, normalize E, pT, eta.

- Replace phi with sin and cos.

- Concatenate global features to each object's features.

So in code:

class MyPreprocessor:
    def __init__(self):
        self.et_miss_mean = None
        self.et_miss_std = None
        self.E_mean = None
        self.E_std = None
        self.pT_mean = None
        self.pT_std = None
        self.eta_mean = None
        self.eta_std = None
        self.obj_id_mapping = None  # maps original obj_id to index (starting from 1)
        self.num_obj_ids = 0

    def fit(self, X, y=None):
        # Compute statistics for E_T_miss and phi_Et_miss
        et_miss = X[:, 0]
        self.et_miss_mean = et_miss.mean().item()
        self.et_miss_std = et_miss.std().item()

        # Collect E, pT, eta from non-padded objects
        # Reshape objects part: (N, 18,5)
        objects = X[:, 2:].view(X.shape[0], 18, 5)
        obj_ids = objects[:, :, 0]
        non_padded_mask = obj_ids != 0

        # Flatten non-padded E, pT, eta
        E = objects[non_padded_mask, 1]
        pT = objects[non_padded_mask, 2]
        eta = objects[non_padded_mask, 3]

        self.E_mean = E.mean().item()
        self.E_std = E.std().item()
        self.pT_mean = pT.mean().item()
        self.pT_std = pT.std().item()
        self.eta_mean = eta.mean().item()
        self.eta_std = eta.std().item()

        # Collect unique obj_ids
        unique_obj_ids = torch.unique(obj_ids[non_padded_mask])
        unique_obj_ids = unique_obj_ids.tolist()
        # Assign indices starting from 1
        self.obj_id_mapping = {id: i+1 for i, id in enumerate(unique_obj_ids)}
        self.num_obj_ids = len(unique_obj_ids)

        return self

    def transform(self, X):
        N = X.shape[0]
        # Process global features
        et_miss = (X[:, 0] - self.et_miss_mean) / self.et_miss_std
        phi_et = X[:, 1]
        sin_phi_et = torch.sin(phi_et)
        cos_phi_et = torch.cos(phi_et)
        global_features = torch.stack([et_miss, sin_phi_et, cos_phi_et], dim=1)  # (N,3)

        # Process objects
        objects = X[:, 2:].view(N, 18, 5)
        obj_ids = objects[:, :, 0]
        E = objects[:, :, 1]
        pT = objects[:, :, 2]
        eta = objects[:, :, 3]
        phi = objects[:, :, 4]

        # Create mask: 1 where obj_id !=0
        mask = (obj_ids != 0).float()  # (N,18)

        # Replace obj_ids with mapped indices
        # Convert to numpy for processing
        obj_ids_np = obj_ids.cpu().numpy()
        mapped_ids = np.zeros_like(obj_ids_np)
        for i in range(N):
            for j in range(18):
                original_id = obj_ids_np[i, j]
                if original_id !=0:
                    mapped_ids[i,j] = self.obj_id_mapping.get(original_id, 0)  # 0 if not found
        mapped_ids = torch.from_numpy(mapped_ids).to(X.device)

        # Normalize E, pT, eta
        E_norm = (E - self.E_mean) / self.E_std
        pT_norm = (pT - self.pT_mean) / self.pT_std
        eta_norm = (eta - self.eta_mean) / self.eta_std

        # Compute sin and cos of phi
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        # Combine object features: [mapped_ids, E_norm, pT_norm, eta_norm, sin_phi, cos_phi]
        obj_features = torch.stack([mapped_ids, E_norm, pT_norm, eta_norm, sin_phi, cos_phi], dim=2)  # (N,18,6)

        # Concatenate global features to each object's features
        global_features_expanded = global_features.unsqueeze(1).expand(-1, 18, -1)  # (N,18,3)
        combined_features = torch.cat([global_features_expanded, obj_features], dim=2)  # (N,18,9)

        # Zero out padded objects
        combined_features = combined_features * mask.unsqueeze(-1)

        return (combined_features, mask)

This preprocessor:

- In fit, computes mean and std for E_T_miss, E, pT, eta.

- Collects unique obj_ids and creates a mapping to indices (starting from 1).

- In transform, processes each event into a sequence of objects with features including global features, and a mask.

Now, the model can be defined as follows:

def make_model(input_shape, *, use_mask=False):
    # input_shape is (18,9) if use_mask is True
    # The model expects a tuple of (X_seq, mask)
    class MyModel(nn.Module):
        def __init__(self, num_obj_ids, embedding_dim=8, d_model=64, nhead=8, num_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(num_embeddings=num_obj_ids +1, embedding_dim=embedding_dim)  # +1 for padding (0)
            self.input_proj = nn.Linear(embedding_dim + 8, d_model)  # 8 = 3 (global) +5 (E, pT, eta, sin_phi, cos_phi)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.classifier = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x, mask=None):
            # x: (combined_features, mask)
            # combined_features: (N,18,9)
            # mask: (N,18)
            combined_features, mask = x
            obj_indices = combined_features[:, :, 3].long()  # Wait, no. The first 3 features are global, then mapped_ids is at index 3?
            # Wait, combined_features is [global_features (3) + obj_features (6)] → total 9.
            # obj_features are [mapped_ids (1), E_norm (1), pT_norm (1), eta_norm (1), sin_phi (1), cos_phi (1)] → 6 features.
            # So combined_features[:, :, 0:3] are global features (et_miss_norm, sin_phi_et, cos_phi_et)
            # combined_features[:, :, 3] is mapped_ids
            # combined_features[:, :, 4] is E_norm
            # combined_features[:, :, 5] is pT_norm
            # combined_features[:, :, 6] is eta_norm
            # combined_features[:, :, 7] is sin_phi
            # combined_features[:, :, 8] is cos_phi

            # So obj_indices is combined_features[:, :, 3].long()
            obj_indices = combined_features[:, :, 3].long()

            # Other features: E_norm, pT_norm, eta_norm, sin_phi, cos_phi → indices 4-8
            other_features = combined_features[:, :, 4:9]  # (N,18,5)

            # Embed obj_indices
            obj_emb = self.embedding(obj_indices)  # (N,18, embedding_dim)

            # Concatenate with other features
            # other_features is (N,18,5)
            # obj_emb is (N,18, embedding_dim)
            # Concatenated: (N,18, embedding_dim +5)
            combined = torch.cat([obj_emb, other_features], dim=2)  # (N,18, embedding_dim+5)

            # Project to d_model
            combined = self.input_proj(combined)  # (N,18, d_model)

            # Transformer expects src_key_padding_mask where True is padding
            src_key_padding_mask = (mask == 0)

            # Transformer
            transformer_out = self.transformer(combined, src_key_padding_mask=src_key_padding_mask)  # (N,18, d_model)

            # Aggregate: mean over non-padded objects
            seq_mask = mask.unsqueeze(-1)  # (N,18,1)
            sum_out = (transformer_out * seq_mask).sum(dim=1)  # (N, d_model)
            count = seq_mask.sum(dim=1)  # (N,1)
            count = torch.clamp(count, min=1e-5)
            mean_out = sum_out / count  # (N, d_model)

            # Classifier
            logit = self.classifier(mean_out).squeeze(1)  # (N,)
            return logit

    # Determine num_obj_ids from the preprocessor
    # But how? The make_model function is called with input_shape, which is determined after preprocessing.
    # Wait, input_shape is passed to make_model as a tuple. If the preprocessor returns (combined_features, mask), then input_shape would be (18,9). But the model needs to know num_obj_ids, which is determined during preprocessing's fit.

    # This is a problem. The make_model function is called with input_shape, but the model needs to know the number of obj_ids to create the embedding layer. However, the preprocessor's fit is called before make_model, so the number of obj_ids is known at the time of model creation.

    # But according to the code template, the make_model function is supposed to create the model based on input_shape and use_mask. So how to pass the num_obj_ids to the model?

    # This suggests that the preprocessor must store the num_obj_ids, and the make_model function must access it. But the code template's make_model function is supposed to be standalone.

    # This indicates a flaw in the approach. The model's architecture depends on the preprocessor's state (num_obj_ids), which is not available when make_model is called.

    # To resolve this, perhaps the preprocessor should be passed to make_model, but according to the code template, make_model is called with input_shape and use_mask.

    # Alternative approach: during preprocessing, the obj_id indices are mapped to integers, and the embedding layer's size is determined by the maximum index found during fit. But this requires the preprocessor to store the number of unique obj_ids plus one (for padding).

    # So in the preprocessor's fit method, self.num_obj_ids = len(unique_obj_ids). Then, when creating the model, the MyPreprocessor instance must be available to retrieve num_obj_ids. But according to the code template, the make_model function is standalone.

    # This suggests that the current approach is not feasible within the code template's constraints.

    # Therefore, perhaps the obj_id should be treated as a continuous feature, but this is not ideal. Alternatively, use a hashing trick for embeddings, but that's not reliable.

    # Alternatively, during preprocessing, the obj_id is one-hot encoded, which would allow the model to process it without needing to know the number of unique obj_ids in advance. But this increases the feature dimension significantly.

    # Given the time constraints, perhaps the best approach is to proceed with the initial plan, assuming that the preprocessor's num_obj_ids is accessible when creating the model. However, according to the code template, the make_model function is called with input_shape and use_mask, which suggests that the model must be created without knowledge of the preprocessor's state.

    # This is a problem. Therefore, perhaps the obj_id should be treated as a continuous feature, normalized similarly to other features. But this is incorrect since obj_id is a categorical variable.

    # Alternatively, during preprocessing, the obj_id is normalized as a continuous feature, but this is not appropriate. However, given the time constraints, perhaps proceed with this approach.

    # Alternatively, use an embedding layer with a sufficiently large num_embeddings. For example, assume that the maximum obj_id is 100, and use an embedding layer with num_embeddings=1000. But this is a hack.

    # Given the time constraints, perhaps proceed with the initial plan, assuming that the preprocessor's num_obj_ids is available when creating the model. However, according to the code template, the make_model function is called before the preprocessor is fit, which is not possible.

    # Therefore, this approach is not feasible. Need to find another way.

    # Alternative plan: treat obj_id as a categorical feature and one-hot encode it during preprocessing. This way, the model doesn't need an embedding layer, and the input dimension is fixed.

    # During preprocessing:

    # Collect all unique obj_ids during fit, create a one-hot encoding.

    # For each object, replace obj_id with a one-hot vector of length K, where K is the number of unique obj_ids.

    # This increases the feature dimension but allows the model to process it without knowing the number of obj_ids in advance.

    # However, this requires storing the unique obj_ids in the preprocessor and one-hot encoding during transform.

    # But this increases the feature dimension significantly. For example, if there are 10 unique obj_ids, each object's features increase by 10-1=9 (assuming one-hot encoding with K-1 features to avoid multicollinearity).

    # But given the code template's constraints, this might be the only way.

    # So modifying the preprocessing steps:

    # In fit:

    # Collect unique obj_ids, create a one-hot encoder.

    # During transform:

    # For each object's obj_id, one-hot encode it.

    # So the transformed object features would be:

    # [one_hot_obj_id, E_norm, pT_norm, eta_norm, sin_phi, cos_phi]

    # The one_hot_obj_id is a vector of length K.

    # This increases the feature dimension per object to K +5.

    # But K can be large, leading to high-dimensional data.

    # However, given the code template's constraints, this might be necessary.

    # So adjusting the preprocessor:

    class MyPreprocessor:
        def __init__(self):
            self.et_miss_mean = None
            self.et_miss_std = None
            self.E_mean = None
            self.E_std = None
            self.pT_mean = None
            self.pT_std = None
            self.eta_mean = None
            self.eta_std = None
            self.obj_categories = None  # list of unique obj_ids

        def fit(self, X, y=None):
            # Compute E_T_miss stats
            self.et_miss_mean = X[:, 0].mean().item()
            self.et_miss_std = X[:, 0].std().item()

            # Process objects
            objects = X[:, 2:].view(X.shape[0], 18, 5)
            obj_ids = objects[:, :, 0]
            non_padded_mask = obj_ids !=0

            # Collect E, pT, eta
            E = objects[non_padded_mask, 1]
            self.E_mean = E.mean().item()
            self.E_std = E.std().item()

            pT = objects[non_padded_mask, 2]
            self.pT_mean = pT.mean().item()
            self.pT_std = pT.std().item()

            eta = objects[non_padded_mask, 3]
            self.eta_mean = eta.mean().item()
            self.eta_std = eta.std().item()

            # Collect unique obj_ids
            self.obj_categories = torch.unique(obj_ids[non_padded_mask]).tolist()
            self.obj_categories = sorted(self.obj_categories)
            self.num_obj_categories = len(self.obj_categories)

            return self

        def transform(self, X):
            N = X.shape[0]
            # Process global features
            et_miss = (X[:, 0] - self.et_miss_mean) / self.et_miss_std
            phi_et = X[:, 1]
            sin_phi_et = torch.sin(phi_et)
            cos_phi_et = torch.cos(phi_et)
            global_features = torch.stack([et_miss, sin_phi_et, cos_phi_et], dim=1)  # (N,3)

            # Process objects
            objects = X[:, 2:].view(N, 18, 5)
            obj_ids = objects[:, :, 0]
            E = objects[:, :, 1]
            pT = objects[:, :, 2]
            eta = objects[:, :, 3]
            phi = objects[:, :, 4]

            # Create mask
            mask = (obj_ids !=0).float()  # (N,18)

            # Normalize E, pT, eta
            E_norm = (E - self.E_mean) / self.E_std
            pT_norm = (pT - self.pT_mean) / self.pT_std
            eta_norm = (eta - self.eta_mean) / self.eta_std

            # Compute sin and cos of phi
            sin_phi = torch.sin(phi)
            cos_phi = torch.cos(phi)

            # One-hot encode obj_ids
            # For each obj_id in obj_ids, create a one-hot vector of length num_obj_categories
            # Convert obj_ids to indices in self.obj_categories
            obj_ids_np = obj_ids.cpu().numpy()
            one_hot = np.zeros((N, 18, self.num_obj_categories), dtype=np.float32)
            for i in range(N):
                for j in range(18):
                    original_id = obj_ids_np[i,j]
                    if original_id !=0:
                        if original_id in self.obj_categories:
                            idx = self.obj_categories.index(original_id)
                            one_hot[i,j,idx] = 1.0
            one_hot = torch.from_numpy(one_hot).to(X.device)

            # Combine object features: [one_hot, E_norm, pT_norm, eta_norm, sin_phi, cos_phi]
            obj_features = torch.cat([
                one_hot,
                E_norm.unsqueeze(-1),
                pT_norm.unsqueeze(-1),
                eta_norm.unsqueeze(-1),
                sin_phi.unsqueeze(-1),
                cos_phi.unsqueeze(-1)
            ], dim=2)  # (N,18, num_obj_categories +5)

            # Concatenate global features to each object's features
            global_features_expanded = global_features.unsqueeze(1).expand(-1,18,-1)  # (N,18,3)
            combined_features = torch.cat([global_features_expanded, obj_features], dim=2)  # (N,18,3 + num_obj_categories +5)

            # Zero out padded objects
            combined_features = combined_features * mask.unsqueeze(-1)

            return (combined_features, mask)

    # Now, the model can process the combined_features as a sequence, with each object's features including the one-hot encoded obj_id, normalized kinematic features, and global features.

    # The model's input_shape would be (18, 3 + num_obj_categories +5). But since num_obj_categories is determined during fit, the input_shape varies based on the data. However, the code template requires make_model to be called with input_shape, which is determined after preprocessing.

    # So in the code template, after preprocessing, the input_shape would be (18, K), where K is 3 + num_obj_categories +5. The make_model function can then create a model that accepts this input_shape.

    # The model can be a transformer-based model that processes the sequence.

    def make_model(input_shape, *, use_mask=False):
        # input_shape is (18, K) where K =3 + num_obj_categories +5
        # use_mask is True
        class MyModel(nn.Module):
            def __init__(self, input_shape):
                super().__init__()
                self.input_shape = input_shape
                self.d_model = 64
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8),
                    num_layers=2
                )
                self.input_proj = nn.Linear(input_shape[-1], self.d_model)
                self.classifier = nn.Sequential(
                    nn.Linear(self.d_model, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )

            def forward(self, x, mask=None):
                # x: (combined_features, mask)
                combined_features, mask = x
                # combined_features: (N,18,K)
                # Project input to d_model
                x = self.input_proj(combined_features)  # (N,18,d_model)

                # Transformer expects src_key_padding_mask where True is padding
                src_key_padding_mask = (mask == 0)

                # Transformer
                x = x.permute(1,0,2)  # (18, N, d_model)
                x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
                x = x.permute(1,0,2)  # (N,18,d_model)

                # Aggregate: mean over non-padded objects
                seq_mask = mask.unsqueeze(-1)  # (N,18,1)
                sum_out = (x * seq_mask).sum(dim=1)  # (N, d_model)
                count = seq_mask.sum(dim=1)
                count = torch.clamp(count, min=1e-5)
                mean_out = sum_out / count

                # Classifier
                logit = self.classifier(mean_out).squeeze(1)
                return logit

        return MyModel(input_shape)

    # This model projects the input features to d_model, processes with a transformer, aggregates, and classifies.

    # However, this approach may not capture the importance of obj_id as well as embeddings. But given the code template's constraints, this might be the best option.

    # Now, implementing the preprocessor and model as described.

    # Training loop:

    # Use BCEWithLogitsLoss, Adam optimizer, learning rate scheduler, early stopping.

    # For the training loop, implement early stopping based on validation loss.

    # Batch size: 256.

    # Epochs: 100 with early stopping patience of 10.

    # Learning rate: 1e-3 with cosine annealing.

    # Also, use class weighting if the dataset is imbalanced. But the problem states that the dataset is roughly 50% signal and 50% background, so weighting may not be necessary.

    # Implementing the training loop:

    def train_model(model, train_loader, val_loader, epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float('inf')
        patience = 10
        no_improve = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            for data, labels in train_loader:
                data, mask = data
                data = data.to(device)
                mask = mask.to(device)
                labels = labels.float().to(device)
                optimizer.zero_grad()
                outputs = model((data, mask))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).long()
                correct += (preds == labels.long()).sum().item()
                total += labels.size(0)
            train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validation
            model.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for data, labels in val_loader:
                    data,

# ----------------  END OF LLM BLOCK ----------------
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    pre = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train) # may be Tensor or Tuple
    X_val   = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    # 2. Build model
    if isinstance(X_train, torch.Tensor):               # single-tensor case
        temp_ref    = X_train
        input_shape = temp_ref.shape[1:]                # e.g. (F,)
        use_mask    = False
    else:                                               # tuple => (data, mask)
        temp_ref    = X_train
        input_shape = temp_ref[0].shape[1:]             # e.g. (L, F)
        use_mask    = True                              
    model = make_model(input_shape, use_mask=use_mask)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy_data = torch.zeros(8, *input_shape, dtype=torch.float32)
        if use_mask:
            toy_mask = torch.zeros(8, input_shape[0], dtype=torch.bool)
            toy_batch = (toy_data, toy_mask)
        else:
            toy_batch = toy_data

        toy_transformed = pre.transform(toy_batch)
        try:
            _ = trained_model(*toy_transformed) if isinstance(toy_transformed, (tuple, list)) \
                else trained_model(toy_transformed)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

    pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
    pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
    pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

    torch.save(trained_model.state_dict(), pth_state)
    with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
    with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

    # 6. Save plots
    _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
    _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 7. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
            "train_loss": tr_loss   if tr_loss else None,
            "val_loss":   va_loss   if va_loss else None,
            "train_acc":  tr_acc    if tr_acc else None,
            "val_acc":    va_acc    if va_acc else None,
        }
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

