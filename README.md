# StreamCore

## Overview

StreamCore is a network resource allocation system for Virtual Network Functions (VNFs) in a network slice. The system uses machine learning models to predict performance metrics and optimize resource allocation for different network components.

## Project Structure

- `main.py`: The main script that demonstrates the complete StreamCore workflow
- `vnf_model.py`: Neural network models for individual VNFs (RAN, OVS, UPF)
- `slice_model.py`: Combines individual VNF models into a network slice model
- `resource_allocation.py`: Implements resource allocation optimization algorithms
- `data_generator.py`: Handles data loading, preprocessing, and normalization

## What the Code Does

StreamCore implements a machine learning-based approach to network resource allocation for network slicing. It focuses on three key Virtual Network Functions (VNFs):

1. **RAN (Radio Access Network)**: Handles wireless access to the network
2. **OVS (Open vSwitch)**: Manages network traffic forwarding
3. **UPF (User Plane Function)**: Processes user data packets

The system:
- Trains neural network models to predict performance metrics for each VNF
- Combines these models to create a network slice model
- Optimizes resource allocation to maximize throughput while meeting QoS requirements
- Evaluates performance with different input parameters

## VNF Model Architecture

StreamCore uses PyTorch neural networks to model each VNF. The models are designed to predict both the mean and standard deviation of performance metrics, enabling regression with uncertainty estimation.

### Model Structure

Each VNF model consists of:
- **Input Layer**: Processes input features (packet size, throughput, resource allocation, etc.)
- **Hidden Layers**: Multiple fully connected layers with ReLU activation functions
- **Output Layers**: Separate layers for mean and standard deviation predictions

### Key Features

- **Uncertainty Estimation**: Models predict both mean values and standard deviations, providing confidence intervals for predictions
- **Negative Log Probability Loss**: Custom loss function that accounts for uncertainty in predictions
- **Adaptive Architecture**: Hidden layer sizes can be customized for each VNF type
- **GPU Acceleration**: Models automatically utilize GPU if available for faster training

### Training Process

The training process includes:
1. **Data Sampling**: Random sampling from training, validation, and test datasets
2. **Batch Processing**: Training on mini-batches for efficiency
3. **Optimization**: Adam optimizer with configurable learning rate and weight decay
4. **Loss Tracking**: Monitoring of training, validation, and test losses
5. **Model Saving**: Periodic saving of model weights and loss history

### Model Customization

Different VNF types can have different model architectures:
- **RAN Model**: Typically uses [64, 32, 16] hidden neurons
- **OVS Model**: Typically uses [64, 32, 16] hidden neurons
- **UPF Model**: Typically uses [32, 16] hidden neurons

## Slice Model Architecture

The Slice Model is a key component of StreamCore that combines individual VNF models to create a comprehensive network slice model. It enables end-to-end performance prediction and resource allocation optimization.

### How It Works

The Slice Model:
1. **Combines VNF Models**: Takes trained models for UPF, OVS, and RAN as inputs
2. **Sequential Processing**: Processes data through each VNF model in sequence
3. **Data Flow**: Passes the output of one VNF as input to the next VNF
4. **Resource Allocation**: Considers resource allocation for each VNF in the slice

### Key Features

- **End-to-End Prediction**: Predicts performance metrics for the entire network slice
- **Throughput Prediction**: Specifically focuses on predicting end-to-end throughput
- **Resource Sensitivity**: Analyzes how different resource allocations affect performance
- **Differentiable Mode**: Supports both differentiable and non-differentiable prediction modes

### Prediction Process

The prediction process follows these steps:
1. **Input Preparation**: Prepares input data for the first VNF (UPF)
2. **Sequential Processing**: Passes data through each VNF model in sequence
3. **Data Transformation**: Normalizes and denormalizes data at each step
4. **Throughput Tracking**: Tracks how throughput changes through the network slice
5. **Final Output**: Returns the final throughput after processing through all VNFs

### Integration with Resource Allocation

The Slice Model is tightly integrated with the resource allocation system:
- It provides the performance prediction needed for resource optimization
- It supports both normalized and denormalized resource values
- It enables the optimization algorithm to find the optimal resource allocation

## What main.py Showcases

The `main.py` script demonstrates the complete StreamCore workflow:

1. **Data Loading**: Loads datasets for RAN, OVS, and UPF from the `net_model_dataset` directory
2. **Model Training**: Trains neural network models for each VNF type
3. **Slice Model Creation**: Combines the individual VNF models into a network slice model
4. **Performance Prediction**: Shows how to predict throughput with given resource allocations
5. **Resource Allocation**: Demonstrates the optimization algorithm for allocating resources to meet QoS requirements

The script includes examples of:
- Training models with different architectures and parameters
- Predicting performance metrics for individual VNFs
- Optimizing resource allocation for the entire network slice
- Handling packet loss and throughput calculations

## Setup Instructions

Create a new python environment:
```bash
sudo apt-get -y install python3-pip
sudo apt-get -y install python3-venv
python3 -m venv ~/myenv
source ~/myenv/bin/activate
```

Install the required python packages:
```bash
pip install -r requirements.txt
```

## Dataset Structure

StreamCore expects datasets in the following directory structure:
```
net_model_dataset/
├── ran/
│   ├── input_dataset.pkl
│   └── output_dataset.pkl
├── ovs/
│   ├── input_dataset.pkl
│   └── output_dataset.pkl
└── upf/
    ├── input_dataset.pkl
    └── output_dataset.pkl
```

Each dataset contains input features (like packet size, throughput, resource allocation) and output features (like packet loss, throughput).

## Usage

Run the main script to see the complete StreamCore workflow:
```bash
python main.py
```

This will:
1. Load the datasets
2. Train the VNF models
3. Create a slice model
4. Perform resource allocation optimization
5. Display the results

## Running without the published testbed dataset

The repository does **not** ship the `net_model_dataset/` pickles, so `main.py`
cannot run out of the box. A schema-correct **synthetic** dataset generator is
provided so the full pipeline is runnable and testable. It is a smoke/integration
dataset (output throughput is a saturating function of offered load and the VNF
resource) — not real testbed measurements.

```bash
python tools/generate_synthetic_dataset.py   # writes net_model_dataset/{ran,ovs,upf}
python tools/smoke_test.py                    # end-to-end: load -> train -> predict -> allocate
```

`smoke_test.py` exits non-zero on failure, so it can gate CI. To use real data,
drop testbed-derived pickles (e.g. `pcap -> dataset`) into `net_model_dataset/`
matching the same schema and skip the generator.

## Uplink direction

`SliceModel.predict_throughput(...)` and `allocate_resources(...)` accept a
`direction` argument:

- `direction='downlink'` (default) — VNF chain in stored order (UPF → OvS → RAN,
  i.e. core → edge → radio).
- `direction='uplink'` — chain reversed (RAN → OvS → UPF, i.e. device → radio →
  core). Uplink is increasingly the congestion point for AI / interactive media
  (contribution / "see-what-I-see"), so the slice must be composable and
  optimisable in this direction.

`res` is always aligned to the stored VNF order regardless of direction.

## MoQ (Media over QUIC) uplink test

```bash
python tools/moq_uplink_test.py
```

Drives the slice with a MoQ-like **uplink** traffic model (object/group cadence,
key-object bursts, uplink-heavy) and exercises the uplink slice path + allocator.
It is a traffic-model simulation — for real packets see the next section.

### Real QUIC transmission

```bash
python tools/moq_quic_transmit.py --seconds 2 --target-mbps 12 --out moq_real.npy
python tools/moq_uplink_test.py --trace moq_real.npy
```

`moq_quic_transmit.py` opens an actual **QUIC** connection (aioquic, loopback)
and pushes MoQ-style media objects **uplink** — one unidirectional stream per
object, key-object bursts per group — with the receiver timestamping every
object. It writes a measured per-window Mbps trace (real QUIC framing / pacing /
congestion control; no sudo or pcap needed). Feed that trace into
`moq_uplink_test.py --trace` to drive the StreamCore uplink allocator from a
real transmission instead of the synthetic model.

### pcap → dataset bridge

```bash
# measured offered-load trace from a real capture
python tools/pcap_to_dataset.py capture.pcap --uplink-src 10.0.0.0/24 \
    --emit moq-trace --out offered.npy
# or write a VNF input dataset (supply --egress-pcap for real output labels)
python tools/pcap_to_dataset.py capture.pcap --emit dataset --vnf ran \
    --out-dir net_model_dataset
```

Converts a testbed packet capture into StreamCore traffic profiles (windowed
packet_size / packet_rate / throughput / inter-arrival stats). A pcap gives the
**offered load** (input features); a VNF transfer function (resource → served
throughput / latency) still needs testbed instrumentation, so single-pcap output
labels are a clearly-marked passthrough placeholder.
