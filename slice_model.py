from scipy.stats import wasserstein_distance
import time
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

pd.set_option('display.float_format', '{:.10f}'.format)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SliceModel:
    def __init__(self, vnf_models, data_gens):
        self.vnf_models = vnf_models
        self.data_gens = data_gens
        
    def predict_slice_data_gen(self, slice_data_gen):
        res_arr = np.vstack([
            slice_data_gen.input_dataset.res_upf.values, 
            slice_data_gen.input_dataset.res_ovs.values, 
            slice_data_gen.input_dataset.res_ran.values
        ]).T
        vnf_output = slice_data_gen.input_dataset.copy()
        
        for vnf_num, (vnf_model, data_gen) in enumerate(zip(self.vnf_models, self.data_gens)):
            # Preparing input data for the current VNF model
            if vnf_num == 0:
                vnf_output = vnf_output.loc[:, data_gen.input_feature_list[:-1]]
                vnf_output['res'] = res_arr[:, vnf_num]
            else:
                res = torch.tensor(res_arr[:, vnf_num].astype(np.float32)).unsqueeze(1).to(device)
                vnf_output = torch.cat((vnf_output, res), dim=1)

            # Normalize, predict, and denormalize through the current VNF model
            vnf_output = data_gen.normalize(vnf_output, feature_type='input')
            vnf_output = vnf_model.predict(vnf_output, mean_val=True)
            vnf_output = data_gen.denormalize(vnf_output, feature_type='output')

        return vnf_output

    def predict_throughput(self, res, input_throughput, differentiable=False, res_normalized=True,
                           verbose=False, direction='downlink'):
        """Predict end-to-end throughput through the slice.

        direction:
            'downlink' (default) processes the VNFs in their stored order
                (UPF -> OvS -> RAN, i.e. core -> edge -> radio).
            'uplink' processes them in reverse (RAN -> OvS -> UPF, i.e.
                device -> radio -> edge -> core). Uplink is increasingly the
                congested direction for AI / interactive media traffic, so the
                slice must be composable in this order.
        `res` is always aligned to the stored VNF order regardless of direction.
        """
        # Ensure 'res' is a list of resource values, denormalizing if needed
        if isinstance(res, torch.Tensor):
            res = res.clone().squeeze()
        n = len(self.vnf_models)
        if res_normalized:
            # Convert each resource in `res` to a denormalized float
            res = [float(self.data_gens[i].denormalize(res[i], feature_type='input', feature='res')) for i in range(n)]

        if direction == 'downlink':
            order = list(range(n))
        elif direction == 'uplink':
            order = list(range(n))[::-1]
        else:
            raise ValueError(f"direction must be 'downlink' or 'uplink', got {direction!r}")

        # Initialize throughput for the first VNF processing
        predicted_output = None
        throughput = input_throughput

        if verbose:
            print(f"[{direction}] Input throughput: {input_throughput}")
        # Iterate through the VNF chain in the order dictated by `direction`
        for step, vnf_num in enumerate(order):
            vnf_model, data_gen = self.vnf_models[vnf_num], self.data_gens[vnf_num]
            # Prepare the input data for this VNF based on its resource allocation and throughput
            input_data = self._prepare_input_data(predicted_output, step == 0, data_gen, throughput, res[vnf_num])

            # Pass the data through normalization, prediction, and denormalization stages
            normalized_input = data_gen.normalize(input_data, feature_type='input')
            predicted_output = vnf_model.predict(normalized_input, mean_val=True)
            denormalized_output = data_gen.denormalize(predicted_output, feature_type='output')

            # Update throughput with the output of this VNF, used as input for the next VNF.
            # Clamp to be non-negative: throughput is a physical quantity and the
            # regression head can otherwise emit small negative values that then
            # propagate (and break the allocator's feasibility logic).
            throughput = torch.clamp(denormalized_output[0, 2], min=0.0)
            if verbose:
                print(f"Throughput after VNF idx {vnf_num} (chain step {step}): {throughput}")
        if verbose:
            print("--------------------------------")
        # Return the final throughput, applying a cap if not differentiable
        if differentiable:
            return torch.clamp(throughput, max=float(input_throughput))
        return max(0.0, min(throughput.item(), input_throughput))


    def _prepare_input_data(self, output_features, is_first, data_gen, throughput, res_val):
        """Helper function to prepare input data for a specific VNF model.

        is_first: True for the first VNF in the chain (builds the feature row
        from the nearest-neighbour traffic profile); False otherwise (chains the
        previous VNF's output and appends this VNF's resource value).
        """
        if is_first:
            # Clamp the offered load to the range the model was trained on; the
            # regression can't extrapolate, and get_nearest_neighbor() returns
            # None outside [min, max] (which previously crashed the slice).
            lo = float(data_gen.input_dataset['throughput'].min())
            hi = float(data_gen.input_dataset['throughput'].max())
            throughput = min(max(float(throughput), lo), hi)
            data_sample = data_gen.get_nearest_neighbor(throughput)
            if data_sample is None:
                return None
            data_sample = [
                data_sample['packet_size'],
                (throughput * 1e6) / (8 * data_sample['packet_size']),
                throughput,
                data_sample['inter_arrival_time_mean'],
                data_sample['inter_arrival_time_std'],
                res_val
            ]
            # dtype=float32 to match the rest of the pipeline (numpy scalars from
            # the nearest neighbour are float64 and would otherwise mismatch on cat).
            return torch.tensor(data_sample, dtype=torch.float).to(device).unsqueeze(0)
        else:
            res_tensor = torch.tensor([res_val], dtype=torch.float).unsqueeze(1).to(device)
            return torch.cat((output_features.to(device), res_tensor), dim=1)
