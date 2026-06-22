from data_generator import DataGenerator as DataGenerator
import os
import torch

from vnf_model import *
from slice_model import *
from resource_allocation import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Fill in the datasets based ont the directory structure.
dataset_folders = {
    "RAN dataset": "./net_model_dataset/ran/",
    "OVS dataset": "./net_model_dataset/ovs/",  
    "UPF dataset": "./net_model_dataset/upf/"
}

# Print dataset filenames and sizes
for dataset_name, dataset_path in dataset_folders.items():
    print(f"{dataset_name}:\n")
    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert bytes to MB
        print(f"  {file} - {file_size:.2f} MB")
    print("\n")

# Load the OvS input and output datasets using the DataGenerator class.
ran_data_gen = DataGenerator(input_dataset_file="./net_model_dataset/ran/input_dataset.pkl",
                              output_dataset_file="./net_model_dataset/ran/output_dataset.pkl", 
                              vnf_type='RAN',
                              norm_type='minmax')
ovs_data_gen = DataGenerator(input_dataset_file="./net_model_dataset/ovs/input_dataset.pkl",
                              output_dataset_file="./net_model_dataset/ovs/output_dataset.pkl", 
                              vnf_type='OvS',
                              norm_type='minmax')
upf_data_gen = DataGenerator(input_dataset_file="./net_model_dataset/upf/input_dataset.pkl",
                              output_dataset_file="./net_model_dataset/upf/output_dataset.pkl", 
                              vnf_type='UPF',
                              norm_type='minmax')

# Fill in the unfinished lines
ran_input_data = ran_data_gen.input_dataset
ran_output_data = ran_data_gen.output_dataset
ovs_input_data = ovs_data_gen.input_dataset
ovs_output_data = ovs_data_gen.output_dataset
upf_input_data = upf_data_gen.input_dataset
upf_output_data = upf_data_gen.output_dataset

print("Dataset loaded successfully!")

# List the input and output features
print(f"RAN Input features: {ran_input_data.columns.tolist()}")
print(f"RAN Output features: {ran_output_data.columns.tolist()}")
print(f"OvS Input features: {ovs_input_data.columns.tolist()}")
print(f"OvS Output features: {ovs_output_data.columns.tolist()}")
print(f"UPF Input features: {upf_input_data.columns.tolist()}")
print(f"UPF Output features: {upf_output_data.columns.tolist()}")

###############################################################################

ran_model = VNF_Model(vnf_typ='ran', 
                      n_inputs=6,
                      n_hidden=[64, 32, 16],
                      n_outputs=5)
ran_model = ran_model.to(device)
train_loss, val_loss = ran_model.fit(ran_data_gen, num_epochs=10000, batch_size=256)
ran_model.plot_loss()
# ran_model.load_weights('./data/saved_weights/ran/model.pth')

ovs_model = VNF_Model(vnf_typ='ovs', 
                      n_inputs=6,
                      n_hidden=[64, 32, 16],
                      n_outputs=5)
ovs_model = ovs_model.to(device)
train_loss, val_loss = ovs_model.fit(ovs_data_gen, num_epochs=10000, batch_size=512)
ovs_model.plot_loss()
# ovs_model.load_weights('./data/saved_weights/ovs/model.pth')

upf_model = VNF_Model(vnf_typ='upf', 
                      n_inputs=6,
                      n_hidden=[32, 16],
                      n_outputs=5)
upf_model = upf_model.to(device)
train_loss, val_loss = upf_model.fit(upf_data_gen, num_epochs=5000, batch_size=512)
upf_model.plot_loss()
# upf_model.load_weights('./data/saved_weights/upf/model.pth')

###############################################################################

# Plotting the RAN model predictions
input_df, _ = ran_data_gen.sample('train')
input_df = torch.tensor(input_df.values, dtype=torch.float).to(device)
pred_df = ran_model.predict(input_df)
pred_df = pred_df.detach().cpu().numpy()

input_throughput = 50
resource_allocation = 2500

try:
    data_sample = ran_data_gen.get_nearest_neighbor(input_throughput)
    if data_sample is None:
        raise ValueError("No data available for the selected values.")

    # Update data sample with input values and normalize
    data_sample = [
        data_sample['packet_size'],
        (input_throughput * 1e6) / (8 * data_sample['packet_size']),
        input_throughput,
        data_sample['inter_arrival_time_mean'],
        data_sample['inter_arrival_time_std'],
        resource_allocation
    ]
    data_sample = ran_data_gen.normalize(np.array([data_sample]), feature_type='input')

    # Model inference
    model_input = torch.tensor(data_sample, dtype=torch.float).to(device).view(1, -1)
    model_output = ran_model(model_input.repeat(2, 1))
    model_output = model_output[0][0].detach().cpu().numpy()
    model_output = ran_data_gen.denormalize(np.array([model_output]), feature_type='output')[0]

    # Calculate output throughput and packet loss
    output_throughput = model_output[2]
    output_throughput = min(input_throughput, output_throughput)
    packet_loss = (input_throughput - output_throughput) / input_throughput

    print(f"Output Throughput: {output_throughput:.2f} Mbps")
    print(f"Packet Loss: {packet_loss:.2%}")
except Exception as e:
    print(f"Error: {e}")

###############################################################################

# Step 1: Create an array of data generators
data_gens = [upf_data_gen, ovs_data_gen, ran_data_gen]
# Step 2: Create an array of VNF models
vnf_models = [upf_model, ovs_model, ran_model]
# Step 3: Create the Slice Model using the VNF models and data generators
slice_model = SliceModel(vnf_models, data_gens)

resource_allocation = {'UPF': 200, #CPU (millicores)
                       'OVS': 20, #Throughput (Mbps)
                       'RAN': 1000} #CPU (millicores)
input_throughput = 35

output_throughput = slice_model.predict_throughput(res = list(resource_allocation.values()),
                                                    input_throughput= input_throughput,
                                                    differentiable=0,
                                                    res_normalized=False,
                                                    verbose=1)
print(f"Output throughput: {output_throughput} Mbps")

###############################################################################

input_throughput = 20 #Mbps
packet_loss = 10 #Percentage

if packet_loss is not None:
    output_throughput = input_throughput * (1 - packet_loss*0.01)
else:
    packet_loss = 0

resource_allocation_norm, qos, _, _, time_taken = allocate_resources(slice_model, input_throughput, output_throughput, verbose=0)
resource_allocation_norm = resource_allocation_norm[0]
resource_allocation = [upf_data_gen.denormalize(resource_allocation_norm[0], feature_type='input', feature='res'), 
                              ovs_data_gen.denormalize(resource_allocation_norm[1], feature_type='input', feature='res'), 
                              ran_data_gen.denormalize(resource_allocation_norm[2], feature_type='input', feature='res')]
resource_allocation = {key: float(value) for key, value in zip(['UPF', 'OVS', 'RAN'], resource_allocation)}
packet_loss = (input_throughput - qos) / input_throughput
print(f"Resource Allocation: {resource_allocation}\n QoS: {qos} Mbps\n Packet Loss: {packet_loss:.2%}\n Time taken: {time_taken:.2f} seconds")
