import numpy as np

import torch
from torch.utils.data import TensorDataset
from int_data import generateInputVectors

import matplotlib.pyplot as plt



def generate_data(dataset_size, toy_params, out_size, auto_balance=False, verbose=True, raw_data=False, device='cpu'):
    """
    Generates a contextual integration task
    """


    MEAN_LIMIT = 0.1875
    TRUE_STEPS = 750 # each step = 1 ms
    # stim_steps = int(np.round(4/5 * toy_params['phrase_length'])) # each step = 1 ms
    stim_steps = int(np.round(1/2 * toy_params['phrase_length'])) # each step = 1 ms


    INPUT_VAR = 1.0/(TRUE_STEPS/stim_steps) # adjust noise variance based on shorter step size to match original paper
    # relax_steps = int(np.round(1/5 * toy_params['phrase_length'])) # number of steps after stimulus where input is inactive
    relax_steps = int(np.round(1/2 * toy_params['phrase_length'])) # number of steps after stimulus where input is inactive

    # Random Gaussian noise for two classes
    color_offset = np.random.uniform(low=-MEAN_LIMIT, high=MEAN_LIMIT, size=dataset_size)
    motion_offset = np.random.uniform(low=-MEAN_LIMIT, high=MEAN_LIMIT, size=dataset_size)
    color_input = np.zeros((dataset_size, stim_steps+relax_steps, 1))
    motion_input = np.zeros((dataset_size, stim_steps+relax_steps, 1))

    # color_input[:, :stim_steps, 0] = np.random.multivariate_normal(color_offset, INPUT_VAR*np.eye(dataset_size), size=stim_steps).T
    # motion_input[:, :stim_steps, 0] = np.random.multivariate_normal(motion_offset, INPUT_VAR*np.eye(dataset_size), size=stim_steps).T
    # Doing it batch by batch is quicker
    for batch_idx in range(dataset_size):
        color_input[batch_idx, :stim_steps, 0] = np.random.normal(color_offset[batch_idx], INPUT_VAR, size=stim_steps)
        motion_input[batch_idx, :stim_steps, 0] = np.random.normal(motion_offset[batch_idx], INPUT_VAR, size=stim_steps)

    # Contextual input, (1, 0) is for color vs. (0, 1) is for motion
    context_input = np.zeros((dataset_size, 1, 2))
    context_input[:, 0, 0] = np.random.randint(2, size=dataset_size)
    context_input[:, 0, 1] = 1 - context_input[:, 0, 0]
    context_input = np.repeat(context_input, repeats=stim_steps+relax_steps, axis=1)

    if toy_params['data_type'] == 'retro_context':
        # Zeros out contextual input until after stim_steps is finished
        delay = stim_steps+5
        delay_end = stim_steps + 15
        if delay is not None:
            context_input[:, :delay, :] = np.zeros((dataset_size, delay, 2))
            context_input[:, delay_end:, :] = np.zeros((dataset_size, relax_steps-15, 2))
            # color_input[:, delay:, 0] = context_input[:, delay:, 0]
            # motion_input[:, delay:, 0] = context_input[:, delay:, 1]

    # Put it all together
    # Final row of context_input serves as EoS trigger
    if toy_params['include_eos']:
        n_inputs = 5
        eos_input = np.zeros((dataset_size, stim_steps+relax_steps, 1))
        eos_input[:, -1, :] = 1.0
        context_input[:, -1, :] = 0.0
        raw_inputs = np.concatenate((
            color_input, motion_input, context_input, eos_input
        ), axis=-1)
    else:
        n_inputs = 4
        raw_inputs = np.concatenate((
            color_input, motion_input, context_inputxw
        ), axis=-1)

    # If needed, maps one hot inputs into corresponding random binary vector forms
    if toy_params['input_type'] == 'one hot':
        toy_params['input_size'] = n_inputs
        inputs = raw_inputs
    elif toy_params['input_type'] in ('binary', 'binary1-1'):
        # Generates word_to_input_vector if its not yet generated
        if 'word_to_input_vector' not in toy_params:
            toy_params['word_to_input_vector'] = generateInputVectors(toy_params)
        inputs = np.matmul(raw_inputs, toy_params['word_to_input_vector']) # (n_batch, n_seq, 4) -> (n_batch, n_seq, n_inputs)
    else:
        raise ValueError('Input type {} not recognized'.format(toy_params['input_type']))

    labels = np.zeros((dataset_size, stim_steps+relax_steps, 1), dtype=np.int32)
    for batch_idx in range(dataset_size):
        context_sum = np.sum(context_input[batch_idx, :, :], axis=0)
        if context_sum[0] > context_sum[1]: # color context
            labels[batch_idx, -1, 0] = int(color_offset[batch_idx] > 0)
        else: # motion context
            labels[batch_idx, -1, 0] = int(motion_offset[batch_idx] > 0)

    masks = np.zeros((dataset_size, stim_steps+relax_steps, 1), dtype=np.int32)
    # masks[:, 0, 0] = 1 # First time step
    masks[:, -1, 0] = 1 # Last time step

    inputs_torch = torch.tensor(inputs, dtype=torch.float, device=device)
    labels_torch = torch.tensor(labels, dtype=torch.long, device=device)
    masks_torch = torch.tensor(masks, dtype=torch.bool, device=device)

    dataset = TensorDataset(inputs_torch, labels_torch)

    if raw_data:
        raw_data = (color_offset, motion_offset, raw_inputs)
        return dataset, masks_torch, raw_data, toy_params
    else:
        return dataset, masks_torch, toy_params

def generate_special_data(toy_params, out_size, device='cpu'):
    """
    Generates special data for the contextual integration task
    """
    
    MEAN_LIMIT = 0.1875
    TRUE_STEPS = 750 # each step = 1 ms
    stim_steps = int(np.round(4/5 * toy_params['phrase_length'])) # each step = 1 ms

    INPUT_VAR = 1.0/(TRUE_STEPS/stim_steps) # adjust noise variance based on shorter step size to match original paper
    relax_steps = int(np.round(1/5 * toy_params['phrase_length'])) # number of steps after stimulus where input is inactive

    N_SPEC_SEQS = 8

    # Random Gaussian noise for two classes
    color_input = np.zeros((N_SPEC_SEQS, stim_steps+relax_steps, 1))
    motion_input = np.zeros((N_SPEC_SEQS, stim_steps+relax_steps, 1))

    # First eight inputs explore the extremes of the average trajectories under both possible contexts
    # (i.e. all possible combinations of color offset, motion offset, and context)
    color_offset = (MEAN_LIMIT, MEAN_LIMIT, -MEAN_LIMIT, -MEAN_LIMIT, MEAN_LIMIT, MEAN_LIMIT, -MEAN_LIMIT, -MEAN_LIMIT)
    motion_offset = (MEAN_LIMIT, -MEAN_LIMIT, MEAN_LIMIT, -MEAN_LIMIT, MEAN_LIMIT, -MEAN_LIMIT, MEAN_LIMIT, -MEAN_LIMIT)

    for batch_idx in range(8): # Just does this using zero variance normals
        color_input[batch_idx, :stim_steps, 0] = np.random.normal(color_offset[batch_idx], 0, size=stim_steps)
        motion_input[batch_idx, :stim_steps, 0] = np.random.normal(motion_offset[batch_idx], 0, size=stim_steps)

    # Contextual input
    context_input = np.zeros((N_SPEC_SEQS, 1, 2))
    context_input[0:4, 0, 0] = 1 # First 4 are one context, next four are other
    context_input[4:8, 0, 0] = 0
    context_input[:, 0, 1] = 1 - context_input[:, 0, 0]
    context_input = np.repeat(context_input, repeats=stim_steps+relax_steps, axis=1)

    # Zeros out contextual input until after stim_steps is finished
    delay = stim_steps+5
    if delay is not None:
        context_input[:, :delay, :] = np.zeros((N_SPEC_SEQS, delay, 2))

    # Final row of context_input serves as EoS trigger
    if toy_params['include_eos']:
        eos_input = np.zeros((N_SPEC_SEQS, stim_steps+relax_steps, 1))
        eos_input[:, -1, :] = 1.0
        context_input[:, -1, :] = 0.0
        inputs = np.concatenate((
            color_input, motion_input, context_input, eos_input
        ), axis=-1)
    else:
        inputs = np.concatenate((
            color_input, motion_input, context_input
        ), axis=-1)

    if toy_params['input_type'] in ('binary', 'binary1-1'):
        inputs = np.matmul(inputs, toy_params['word_to_input_vector']) # (n_batch, n_seq, 4) -> (n_batch, n_seq, n_inputs)

    labels = np.zeros((N_SPEC_SEQS, stim_steps+relax_steps, 1), dtype=np.int32)
    for batch_idx in range(N_SPEC_SEQS):
        context_sum = np.sum(context_input[batch_idx, :, :], axis=0)
        if context_sum[0] > context_sum[1]: # color context
            labels[batch_idx, -1, 0] = int(color_offset[batch_idx] > 0)
        else: # motion context
            labels[batch_idx, -1, 0] = int(motion_offset[batch_idx] > 0)

    masks = np.zeros((N_SPEC_SEQS, stim_steps+relax_steps, 1), dtype=np.int32)
    # masks[:, 0, 0] = 1 # First time step
    masks[:, -1, 0] = 1 # Last time step

    inputs_torch = torch.tensor(inputs, dtype=torch.float, device=device)
    labels_torch = torch.tensor(labels, dtype=torch.long, device=device)
    masks_torch = torch.tensor(masks, dtype=torch.bool, device=device)

    dataset = TensorDataset(inputs_torch, labels_torch)

    return dataset, masks_torch

def generate_cont_int_data(dataset_size, toy_params, out_size, auto_balance=False, verbose=True, raw_data=False, device='cpu'):
    """
    Generates a continuous data integration task
    """

    MEAN_LIMIT = 0.1875
    TRUE_STEPS = 750 # each step = 1 ms
    assert toy_params['n_classes'] == 2
    assert toy_params['include_eos']
    stim_steps = toy_params['phrase_length'] # each step = 1 ms


    INPUT_VAR = 1.0/(TRUE_STEPS/stim_steps) # adjust noise variance based on shorter step size to match original paper

    # Random Gaussian noise for two classes
    offset = np.random.uniform(low=-MEAN_LIMIT, high=MEAN_LIMIT, size=dataset_size)
    cont_input = np.zeros((dataset_size, stim_steps, 1))

    # Doing it batch by batch is quicker
    for batch_idx in range(dataset_size):
        cont_input[batch_idx, :stim_steps, 0] = np.random.normal(offset[batch_idx], INPUT_VAR, size=stim_steps)

    raw_inputs = cont_input
    # Maps one hot inputs into corresponding random binary vector forms
    if toy_params['input_type'] == 'one hot':
        raise NotImplementedError()
    elif toy_params['input_type'] in ('binary', 'binary1-1'):
        # Generates word_to_input_vector if its not yet generated
        if 'word_to_input_vector' not in toy_params:
            toy_params['words'] = ['evid0', '<eos>',]
            toy_params['word_to_input_vector'] = generateInputVectors(toy_params)
        # Uses 'evid0' as the continuous input
        cont_input = np.matmul(cont_input, toy_params['word_to_input_vector']['evid0'][np.newaxis, :]) # (n_batch, n_seq, 1) -> (n_batch, n_seq, n_inputs)
    else:
        raise ValueError('Input type {} not recognized'.format(toy_params['input_type']))

    # Adds the EoS trigger
    cont_input[:, -1, :] = toy_params['word_to_input_vector']['<eos>']

    labels = np.zeros((dataset_size, stim_steps, 1), dtype=np.int32)
    for batch_idx in range(dataset_size):
        labels[batch_idx, -1, 0] = int(offset[batch_idx] > 0)

    masks = np.zeros((dataset_size, stim_steps, 1), dtype=np.int32)
    # masks[:, 0, 0] = 1 # First time step
    masks[:, -1, 0] = 1 # Last time step

    inputs_torch = torch.tensor(cont_input, dtype=torch.float, device=device)
    labels_torch = torch.tensor(labels, dtype=torch.long, device=device)
    masks_torch = torch.tensor(masks, dtype=torch.bool, device=device)

    dataset = TensorDataset(inputs_torch, labels_torch)

    if raw_data:
        raw_data = (raw_inputs, offset,)
        return dataset, masks_torch, raw_data, toy_params
    else:
        return dataset, masks_torch, toy_params