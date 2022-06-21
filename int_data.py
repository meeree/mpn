import numpy as np

import time
import copy

import torch
from torch.utils.data import TensorDataset


def generate_word_bank(toy_params):
    """ Creates the word bank based on various inputs """

    variable_length = toy_params['variable_length'] if 'variable_length' in toy_params else False

    if toy_params['ordered_classes']:
        words = ['awful', 'bad', 'good', 'awesome']
    else:
        words = []
        for i in range(toy_params['n_classes']):
            words.extend(['evid'+str(i)])
    
    words.extend(['null'])

    if toy_params['context_words'] is not None:
        words.extend(toy_params['context_words'])
    if toy_params['include_eos']:
        words.append('<eos>')
    if toy_params['include_sos']:
        words.append('<sos>')
    if toy_params['n_delay'] != 0 :
        if toy_params['delay_word'] == '<delay>':
            words.append('<delay>')
        elif toy_params['delay_word'] != 'null':
            raise ValueError('Delay word not recognized.')

    return words

def make_toy_phrase_rand_words(toy_params, phrase_length):
    """ 
    Creates a single toy phrase (in word form, not tensor form) by randomly drawing from the word bank, 
    with some additional rules for certain context words. NOTE: For uniform score generation this is not used.

    This encodes special rules for the frequency of certain context words.

    All special characters like <eos>, <sos>, or <pads> are handled externally and the corresponding 
    adjustment of phrase_length are handled externally.

    """
    
    words = toy_params['words']
    variable_length = toy_params['variable_length']
    min_phrase_len = toy_params['min_phrase_len']

    phrase = []

    n_words = len(words)

    for idx in range(phrase_length):
        next_word = False
        
        while not next_word:
            next_word = False

            word_idx = np.random.randint(n_words)
            if words[word_idx] == 'extremely' and idx > 0: # No repeat 'extremely'
                if phrase[idx-1] != 'extremely':
                    next_word = True
            elif words[word_idx] == 'not' and idx > 0: # No repeat 'not'
                if phrase[idx-1] != 'not':
                    next_word = True
            elif words[word_idx] == 'not4' and idx > 0: # No repeat 'not4'
                found_not = False
                for idx2 in range(1, min(4, idx+1)): # Up to 4 words back
                    if phrase[idx-idx2]== 'not4':
                        found_not = True
                if not found_not:
                    next_word = True
            elif words[word_idx] == 'not8' and idx > 0: # No repeat 'not8'
                found_not = False
                for idx2 in range(1, min(8, idx+1)): # Up to 8 words back
                    if phrase[idx-idx2] == 'not8':
                        found_not = True
                if not found_not:
                    next_word = True
            elif words[word_idx] not in ('<eos>', '<sos>', '<pad>'):
                next_word = True
        phrase.append(words[word_idx])
    
    return phrase

def base_word_values(toy_params):
    """ 
    Generates the base word values which are used to score phrases.

    For ordered classes this is just a single number.

    For unordered classes (default case) this is an N-dimensional vector wiht
    N the number of classes. 

    """

    if toy_params['ordered_classes']:
        base_word_vals = {'awful': -2.0, 'bad': -1.0, 'null': 0, 'good': 1.0, 'awesome': 2.0}
    else:
      n_classes = toy_params['n_classes']
      base_word_vals = {}
      for i in range(n_classes):
          base_word_vals['evid'+str(i)] = np.zeros((n_classes,))
          base_word_vals['evid'+str(i)][i] = 1

    return base_word_vals

def score_toy_phrase(toy_phrase, toy_params):
    """ 
    Scores a single toy phrase.

    This is calculated using the base_word_vals assigned to each input word as well
    as special rules assigned to certain contextual words. 

    This function encodes the definition of how certain contextual words
    modify a given sequence.

    """
    base_word_vals = toy_params['base_word_vals']

    phrase_length = len(toy_phrase)
    extreme_length = 0  # current range of influence of extreme
    not_length = 0      # current range of influence of not

    if toy_params['ordered_classes']: # ordered score
        score = 0
    else:
        score = np.zeros((toy_params['n_classes'],))

    for idx in range(phrase_length):
        if toy_phrase[idx] in list(base_word_vals.keys()):
            base_score = base_word_vals[toy_phrase[idx]]
            if not_length > 0: 
                base_score = -1 * base_score
            if extreme_length > 0: 
                base_score = 2 * base_score
            score += base_score
        elif toy_phrase[idx] == 'not':
            not_length = 2
        elif toy_phrase[idx] == 'not4':
            not_length = 4
        elif toy_phrase[idx] == 'not8':
            not_length = 8
        elif toy_phrase[idx] == 'extremely':
            extreme_length = 2
        # elif toy_phrase[idx] == '<EoS>': break
        # elif toy_phrase[idx] == '<pad>': break
        
        if not_length > 0: not_length -= 1          # decays not
        if extreme_length > 0: extreme_length -= 1  # decays extremely

    return score

def classifySentiment(score, toy_params):
    """
    Turns a score of a toy phrase into a sentiment and corresponding tensor
    
    Contains definitions of thresholds for dividing into multiple classes.

    Note this also encodes how certain ties in scores are broken (usually via an argmax function),
    but does not contain code to reject ties.
    
    """
    sentiments = toy_params['classes']
    ordered = toy_params['ordered_classes']
    loss_type = toy_params['loss_type']
    phrase_length = toy_params['phrase_length']
    outer_prod = toy_params['outer_prod']

    n_sentiments = len(sentiments)
    n_outputs = toy_params['net_out_size']

    if outer_prod:
        sentiment = []
        sentiment_tensor_final = np.zeros((n_outputs,))
        for score_idx in range(score.shape[0]):
            if score[score_idx] >= 0:
                sentiment.append('Pos')
                sentiment_tensor_final[score_idx] = 1.0
            else:
                sentiment.append('Neg')
        # This is a hack that just sets all times of sentiment tensor equal to final
        sentiment_tensor = np.array([sentiment_tensor_final for idx in range(phrase_length)])
    elif not ordered:  # Usual unordered (uniform over words, not scores)
        sentiment = 'class'+str(np.argmax(score))
        sentiment_tensor_final = np.zeros((n_outputs,))
        sentiment_tensor_final[np.argmax(score)] = 1.0
        # This is a hack that just sets all times of sentiment tensor equal to final
        sentiment_tensor = np.array([sentiment_tensor_final for idx in range(phrase_length)])
    elif ordered and toy_params['uniform_score']: # Ordered and uniform score
      
        # Automatically subdivide possible scores based on number of classes
        sub_divs = []
        if len(toy_params['words']) == 3:
            score_range = 2*phrase_length + 1
            for class_idx in range(1, n_sentiments):
                sub_divs.append(-1*phrase_length + class_idx*score_range/n_sentiments)
        elif len(toy_params['words']) == 5:
            score_range = 4*phrase_length + 1
            for class_idx in range(1, n_sentiments):
                sub_divs.append(-2*phrase_length + class_idx*score_range/n_sentiments)

        class_found = False
        class_idx = 0
        while not class_found:
            if class_idx == n_sentiments - 1: # Checked all other classes
                class_found = True
            elif score <= sub_divs[class_idx]:
                class_found = True
            else:
                class_idx += 1
          
        sentiment = sentiments[class_idx]

        sentiment_tensor_final = np.zeros((n_outputs,))
        sentiment_tensor_final[sentiments.index(sentiment)] = 1.0
        # This is a hack that just sets all times of sentiment tensor equal to final
        sentiment_tensor = np.array([sentiment_tensor_final for idx in range(phrase_length)])

    else: # Ordered class examples (no uniform score)
    
        if toy_params['uniform_score']:
            # Used for 3-class sentiment analysis, about 1/3 will be neutral
            neutral_thresh = 1/3 * phrase_length

            # 5-class sentiment analysis, about 1/5 will be of each class
            three_star_thresh = 1/5 * phrase_length
            four_star_thresh = 3/5 * phrase_length
        else:
            neutral_thresh = 1/3 * phrase_length

            three_star_thresh = 0.05 * phrase_length
            four_star_thresh = 0.18 * phrase_length

        if loss_type == 'XE':
            if n_sentiments == 2:
                if score >= 0:
                    sentiment = 'Good'
                else:
                    sentiment = 'Bad'
            elif n_sentiments == 3:
                if score >= neutral_thresh:
                    sentiment = 'Good'
                elif score <= -1 * neutral_thresh:
                    sentiment = 'Bad'
                else:
                    sentiment = 'Neutral'
            elif n_sentiments == 5:
                if score > four_star_thresh:
                    sentiment = 'Five'
                elif score > three_star_thresh:
                    sentiment = 'Four'
                elif score > -1 * three_star_thresh:
                    sentiment = 'Three'
                elif score > -1 * four_star_thresh:
                    sentiment = 'Two'
                else:
                    sentiment = 'One'
            else:
                raise NotImplementedError('n_sentiments only implemented for 2, 3, or 5 class')
            sentiment_tensor_final = np.zeros((n_outputs,))
            sentiment_tensor_final[sentiments.index(sentiment)] = 1.0
            # This is a hack that just sets all times of sentiment tensor equal to final
            sentiment_tensor = np.array([sentiment_tensor_final for idx in range(phrase_length)])
        elif loss_type == 'MSE':
            sentiment = score
            sentiment_tensor = np.array([[0.0] if idx != phrase_length-1 else [score] for idx in range(phrase_length)])
    
    return sentiment, sentiment_tensor

def wordToIndex(word, word_bank):
    """ Converts a word into corresponding index in words """
    return word_bank.index(word)

def wordToTensor(word, word_bank):
    """ Turn a letter into a <1 x n_words> Tensor """
    n_words = len(word_bank)
    tensor = np.zeros((1, n_words))
    tensor[0][wordToIndex(word, word_bank)] = 1
    return np.array(tensor)

def phraseToTensor(phrase, toy_params):
    """ Turn a phrase into a <phrase_length x input_size>  """
    
    word_bank = toy_params['words']

    tensor = np.zeros((len(phrase), toy_params['input_size']))
    for word_idx, word in enumerate(phrase):
        tensor[word_idx] = toy_params['word_to_input_vector'][word]

    return np.array(tensor)

def generateInputVectors(toy_params):
    """
    
    This creates the word_to_input_vector that is the one-to-one mapping between a given
    word (in string form) to its corresponding input vector. 

    This is used for all types of possible inputs, including one-hot and random binary.

    """

    # word_to_input_vector = np.zeros((len(toy_params['words']), toy_params['input_size']))
    word_to_input_vector = {}

    for word_idx, word in enumerate(toy_params['words']):
        vector_found = False
        while not vector_found: 
            if toy_params['input_type'] in ('binary', 'binary1-1', 'binary_no_norm'):
                candidate_vec = np.random.randint(2, size=(toy_params['input_size'],))
                candidate_vec = np.array(candidate_vec, dtype=float)
                # Normalized to expected unit magnitude (same norm for all helps with analytics, but isn't needed)
                # Note this is skipped for binary_no_norm 
                if toy_params['input_type'] == 'binary1-1':
                    candidate_vec = 2 * candidate_vec - 1 # Changes to -1 and 1 values
                    candidate_vec /= np.sqrt(toy_params['input_size'])
                elif toy_params['input_type'] == 'binary':
                    candidate_vec /= np.sqrt(1/2*toy_params['input_size'])
            elif toy_params['input_type'] in ('one_hot',):
                candidate_vec = np.zeros((toy_params['input_size'],))
                candidate_vec[word_idx] = 1.0
            else:
                raise ValueError('Input type {} not recognized'.format(toy_params['input_type']))
            # Checks to see if vector representation already exists
            repeat_found = False
            for word_idx2 in range(word_idx):
                if (candidate_vec == word_to_input_vector[toy_params['words'][word_idx2]]).all():
                    repeat_found = True

            vector_found = False if repeat_found else True

        word_to_input_vector[word] = candidate_vec

    return word_to_input_vector

def tensorToPhrase(tensor, word_bank):
    """ Turn an array of one-hot letter vectors into a phrase """
    phrase = []
    for idx in range(tensor.shape[0]):
        hot_idx = np.argmax(tensor[idx])
        phrase.append(word_bank[hot_idx])
    return phrase

def randomTrainingExample(toy_params, ):
    """
    Generates a random training example consisting of phrase and sentiment and corresponding tensors

    Returns:
    sentiment_tensor: time x input dim (word bank size)
    sentiment_tensor: 1 x output dim (sentiment bank size)
    target_mask:

    """
    # Unpacks toy_params
    max_phrase_length = np.copy(toy_params['phrase_length'])
    words = toy_params['words']
    sentiments = toy_params['classes']
    
    loss_type = toy_params['loss_type']
    variable_length = toy_params['variable_length'] 

    uniform_score = toy_params['uniform_score']
    ordered_class = toy_params['ordered_classes'] 

    # If variable length is allowed, randomly generates phrase length
    if variable_length:
        min_phrase_len = toy_params['min_phrase_len'] 
        phrase_length = min_phrase_len + np.random.randint(max_phrase_length - min_phrase_len)
    else:
        phrase_length = np.copy(max_phrase_length)

    ### Randomly generates just the phrase (in words, converted to tensors later) under many conditions ###

    # generates shorter phrases if special characters are included
    # note this means phrase_length includes the special character count
    if toy_params['include_eos']:
        phrase_length = phrase_length - 1
    if toy_params['include_sos']:
        phrase_length = phrase_length - 1
    if toy_params['n_delay'] != 0:
        phrase_length = phrase_length - np.abs(toy_params['n_delay'])

    if  phrase_length <= 0:
        raise ValueError('Phrase length is less than zero!')

    if uniform_score and not ordered_class: # Uniform score generation for unordered classes

        n_scores_phrase = n_scores(len(sentiments), phrase_length)
        valid_score = False
        while not valid_score:
            score_idx = np.random.randint(n_scores_phrase)
            score = index_to_score_unordered(score_idx, len(sentiments), phrase_length)
            # If ties are eliminated, makes sure maximum only occurs once
            if np.sum(np.max(score) == score) == 1 or not toy_params['eliminate_ties']:
                valid_score = True
            # else:
            #     print('Rejected score:', score)

        # print('Phrase length 2:', phrase_length)
        phrase = score_to_phrase_unordered(score, phrase_length, toy_params)
            
    elif uniform_score and ordered_class: # Uniform score generation for ordered classes
        if len(words) == 3: # good, bad, the case
            n_scores_phrase = 2*phrase_length+1
            score_idx = np.random.randint(n_scores_phrase)
            score = index_to_score_ordered(score_idx, phrase_length)
            phrase = score_to_phrase_ordered(score, toy_params)
        else: # more general case
            score_to_idx_map = toy_params['score_to_idx_map']
            score_vals = list(score_to_idx_map.keys())
            score_idx = np.random.randint(len(score_to_idx_map)) # random score index
            score = score_vals[score_idx]
            phrase = score_to_phrase_ordered_general(score, score_to_idx_map, toy_params)
    else: # Default is to just generate phrase by randomly drawing from word bank, then score afterwards.
        phrase = make_toy_phrase_rand_words(toy_params, phrase_length)
        # Note score is not dependent upon special characters added below, so can calculate before they are added
        score = score_toy_phrase(phrase, toy_params)

    if toy_params['include_sos']:
        phrase.insert(0, '<sos>')
    if toy_params['n_delay'] > 0: # Insert delay at end (before <eos> though)
        phrase.extend([toy_params['delay_word'] for _ in range(toy_params['n_delay'])])
    elif toy_params['n_delay'] < 0: # Inserts delay at the start instead
        phrase[0:0] = [toy_params['delay_word'] for _ in range(abs(toy_params['n_delay']))]
    if toy_params['include_eos']:
        phrase.append('<eos>')

    length = len(phrase)
    if variable_length:
        if length < max_phrase_length:
            phrase.extend(['<pad>' for _ in range(max_phrase_length-length)])
    target_mask = np.array([length-1], dtype=np.int32) # When target is defined.
    
    sentiment, sentiment_tensor = classifySentiment(score, toy_params)
    phrase_tensor = phraseToTensor(phrase, toy_params)
    
    return sentiment, phrase, sentiment_tensor, phrase_tensor, target_mask

def generate_data(dataset_size, toy_params, out_size, auto_balance=False, verbose=True, raw_data=False, create_tensors=True, device='cpu'):
    """
    Generate training data in numpy
    """

    #### Sets default parameter values if they aren't already set ####
    if 'loss_type' not in toy_params: toy_params['loss_type']  = 'XE'
    if 'ordered_classes' not in toy_params: toy_params['ordered_classes']  = False
    if 'uniform_score' not in toy_params: toy_params['uniform_score']  = False
    if toy_params['uniform_score']:
        if 'eliminate_ties' not in toy_params: toy_params['eliminate_ties'] = True
    else: # Eliminate ties not yet implemented for non-uniform scores
        if 'eliminate_ties' not in toy_params: toy_params['eliminate_ties'] = False
    if 'filter_classes' not in toy_params: toy_params['filter_classes']  = False
    if 'input_type' not in toy_params: toy_params['input_type']  = 'one hot'
    if 'noise_var' not in toy_params: toy_params['noise_var']  = None
    # Optional ability to stack multiple phrases end to end
    if 'stack_phrases' not in toy_params: toy_params['stack_phrases'] = False
    if 'n_stack' not in toy_params: toy_params['n_stack'] = 1
    # Special characters at start or end of phrase
    if 'include_eos' not in toy_params: toy_params['include_eos'] = False
    if 'include_sos' not in toy_params: toy_params['include_sos'] = False
    # Insertion of delay period
    if 'n_delay' not in toy_params: toy_params['n_delay'] = 0
    if 'delay_word' not in toy_params: toy_params['delay_word'] = '<delay>' # '<delay>' or 'null'


    if 'context_words' not in toy_params: toy_params['context_words'] = None
    if 'outer_prod' not in toy_params: toy_params['outer_prod'] = False
    if 'variable_length' not in toy_params: toy_params['variable_length']  = False
    if 'min_phrase_len' not in toy_params: toy_params['min_phrase_len']  = 0
    if 'label_shift' not in toy_params: toy_params['label_shift']  = None

    if 'words' not in toy_params:
        toy_params['words'] = generate_word_bank(toy_params)
    if 'base_word_vals' not in toy_params: # Values of each words in terms of their scores
        toy_params['base_word_vals'] = base_word_values(toy_params)
    if  'word_to_input_vector' not in toy_params: # Creates a unique input vector for each word
        if toy_params['input_type'] in ('one_hot', 'binary', 'binary1-1', 'binary_no_norm'):
            toy_params['word_to_input_vector'] = generateInputVectors(toy_params)
            if '<delay>' in toy_params['words']: # Make the delay input vector smaller
                if 'delay_scale' not in toy_params: toy_params['delay_scale']  = 0.0
                toy_params['word_to_input_vector']['<delay>'] = toy_params['delay_scale'] * toy_params['word_to_input_vector']['<delay>']
        else:
            raise ValueError('Input type not recognized!')

    if 'input_size' not in toy_params: toy_params['input_size']  = len(toy_params['words'])

    if 'classes' not in toy_params:
        if toy_params['ordered_classes']:
            raise NotImplementedError()
        else:
            toy_params['classes'] = ['class'+str(i) for i in range(toy_params['n_classes'])]

    toy_params['net_out_size'] = out_size # Controls size of sentiment tensors

    uniform_score = toy_params['uniform_score']
    filter_classes = toy_params['filter_classes'] 
    ordered_class = toy_params['ordered_classes'] 

    if toy_params['stack_phrases']: # Modifies dataset size so it generates enough for stacking later
        n_stacked_datasets = np.copy(dataset_size) # original number to be used later
        dataset_size = int(dataset_size * toy_params['n_stack'])

    syn_sentiments = []
    syn_phrases = []
    syn_inputs_np = np.zeros((dataset_size, toy_params['phrase_length'], toy_params['input_size'],))
    syn_targets_np = np.zeros((dataset_size, toy_params['phrase_length'], out_size,), dtype=np.int32)
    syn_target_masks_np = np.zeros((dataset_size, toy_params['phrase_length'], out_size,), dtype=np.int32)
    # syn_target_masks_np = []

    # print('syn_targets_np', syn_targets_np.shape)

    outer_prod = toy_params['outer_prod'] if 'outer_prod' in toy_params else False

    # Checks for uniform score implementation:
    if uniform_score and toy_params['ordered_classes']:
        check_word_bank_for_uniform_score_ordered(toy_params['words'])

        if len(toy_params['words']) == 5 or len(toy_params['words']) == 6:
            if 'score_to_idx_map' not in toy_params:
                score_vals = toy_params['score_vals'] if 'score_vals' in toy_params else [-2, -1, 1, 2]
                toy_params['score_to_idx_map'] = generate_score_map(len(score_vals), toy_params['phrase_length'], score_vals)
    elif uniform_score: # Uniform score unordered
        check_word_bank_for_uniform_score(toy_params['words'], len(toy_params['classes']))

    start_time = time.time()
    if auto_balance: # Ensures the classes are balanced
        if outer_prod:
            n_classes = 2**len(toy_params['classes'])
        else:
            n_classes = len(toy_params['classes'])
        if filter_classes:
            class_filter_idxs = []
            print('Filtering out classes:', class_filter_idxs)
            num_per_class = np.ones((n_classes,))*int(dataset_size/(n_classes- len(class_filter_idxs)))
            if class_filter_idxs != []:
                num_per_class[tuple(class_filter_idxs)] = 0
        else:
            num_per_class = np.ones((n_classes,))*int(dataset_size/n_classes)
        print('Looking for num per class:', num_per_class)
        class_count = np.zeros((n_classes,))
        complete = False
        while not complete:
            sentiment, phrase, sentiment_tensor, phrase_tensor, target_mask = randomTrainingExample(toy_params)
            if outer_prod: # Treats sentiment like a binary number and converts it into a class index
                class_idx = int(np.sum([sentiment_tensor[0, sent_idx]*(2**sent_idx) for sent_idx in range(sentiment_tensor.shape[1])]))
            else:
                class_idx = np.argmax(sentiment_tensor[0])

            if class_count[class_idx] < num_per_class[class_idx]:
                trial = int(np.sum(class_count))
                class_count[class_idx] += 1

                syn_targets_np[trial, :, :] = sentiment_tensor
                syn_inputs_np[trial, :, :] = phrase_tensor
                if len(target_mask) > 1:
                    raise NotImplementedError
                syn_target_masks_np[trial, target_mask, :] = np.ones((out_size,), dtype=np.int32)

                if (class_count == num_per_class).all(): # Checks to see if finished
                    complete = True
    else:
        for trial in range(dataset_size):
            sentiment, phrase, sentiment_tensor, phrase_tensor, target_mask = randomTrainingExample(toy_params)
            
            syn_targets_np[trial, :, :] = sentiment_tensor
            syn_inputs_np[trial, :, :] = phrase_tensor
            syn_phrases.append(phrase)
            if len(target_mask) > 1:
                raise NotImplementedError
            syn_target_masks_np[trial, target_mask, :] = np.ones((out_size,), dtype=np.int32)

    if verbose:
        print('Synthentic data generated in: {:0.2f} sec. Autobalanced: {}. Uniform score: {}. Eliminate ties: {}'.format(
            time.time() - start_time, auto_balance, uniform_score, toy_params['eliminate_ties']))

    # syn_inputs_np = np.transpose(syn_inputs_np, axes=(1, 0, 2))             # Phrase tensors: dataset_size x phrase_len x in_dim -> phrase_len, dataset_size, in_dim
    # syn_targets_np = np.transpose(syn_targets_np, axes=(1, 0, 2))           # Sentiment tensors: dataset_size x phrase_len x out_dim -> phrase_len, dataset_size, out_dim
    # syn_target_masks_np = np.transpose(syn_target_masks_np, axes=(1, 0, 2)) # Target mask: dataset_size x phrase_len x out_dim -> phrase_len, dataset_size, out_dim

    syn_targets_np = np.argmax(syn_targets_np, axis=-1)[:, :, np.newaxis] # Compresses last dimension to max argument

    # print('Sample input:', syn_inputs_np[:21, 0, :])
    # print('Sample targets:', syn_targets_np[:21, 0, :])

    # Stacks phrases end to end to create longer temporal streams. Also stacks masks and targets.
    if toy_params['stack_phrases']:
        if toy_params['variable_length']:
            raise NotImplementedError()

        phrase_len = toy_params['phrase_length']
       
        syn_inputs_stack = np.zeros((n_stacked_datasets, phrase_len*toy_params['n_stack'], toy_params['input_size'],))
        syn_targets_stack = np.zeros((n_stacked_datasets, phrase_len*toy_params['n_stack'], 1,))
        syn_target_masks_stack  = np.zeros((n_stacked_datasets, phrase_len*toy_params['n_stack'], out_size,))

        stack_count = 0
        batch_count = 0
        for data_idx in range(dataset_size):
            syn_inputs_stack[batch_count, stack_count*phrase_len:(stack_count+1)*phrase_len, :] = syn_inputs_np[data_idx, :, :]
            syn_targets_stack[batch_count, stack_count*phrase_len:(stack_count+1)*phrase_len, :] = syn_targets_np[data_idx, :, :]
            syn_target_masks_stack[batch_count, stack_count*phrase_len:(stack_count+1)*phrase_len, :] = syn_target_masks_np[data_idx, :, :]
            stack_count += 1
            if stack_count == toy_params['n_stack']:
                stack_count = 0
                batch_count += 1
        syn_inputs_np = syn_inputs_stack
        syn_targets_np =syn_targets_stack
        syn_target_masks_np = syn_target_masks_stack

        # syn_inputs_np = np.reshape(syn_inputs_np, (phrase_len*toy_params['n_stack'], n_stacked_datasets, len(toy_params['words'])))
        # syn_targets_np = np.reshape(syn_targets_np, (phrase_len*toy_params['n_stack'], n_stacked_datasets, 1))
        # syn_target_masks_np = np.reshape(syn_target_masks_np, (phrase_len*toy_params['n_stack'], n_stacked_datasets, out_size))

    # Optional shift of targets (used for cata forgetting code to generate distinct classes)
    if toy_params['label_shift'] is not None: 
        syn_targets_np = syn_targets_np + toy_params['label_shift']

    if create_tensors: # Turns everything into Pytorch tensors, and creates a dataset
        # Lastly, converts to pyTorch tensors
        syn_inputs = torch.tensor(syn_inputs_np, dtype=torch.float, device=device)
        syn_targets = torch.tensor(syn_targets_np, dtype=torch.long, device=device)
        syn_target_masks = torch.tensor(syn_target_masks_np, dtype=torch.bool, device=device)

        if toy_params['noise_var'] is not None:
            noise_means = torch.zeros_like(syn_inputs)
            noise_vars = toy_params['noise_var'] * torch.ones_like(syn_inputs)
            syn_inputs = syn_inputs + torch.normal(noise_means, noise_vars)

        # print('Input shape:', syn_inputs.shape)
        # print('Target shape:', syn_targets.shape)

        dataset = TensorDataset(syn_inputs, syn_targets)

        # Note this passes back toy_params because it updates it can be updated internally and may want to be used elsewhere
        if raw_data:
            raw_data = (syn_phrases,)
            return dataset, syn_target_masks, raw_data, toy_params
        else:
            return dataset, syn_target_masks, toy_params
    else: # Just returns numpy version
        if raw_data:
            raw_data = (syn_phrases,)
            return syn_inputs_np, syn_targets_np, syn_target_masks_np, raw_data, toy_params
        else:
            return syn_inputs_np, syn_targets_np, syn_target_masks_np, toy_params


def generate_special_data(toy_params, out_size, device='cpu'):
    """
    Generate special test data in numpy 
    """

    word_bank = toy_params['words']

    syn_inputs_np = []
    syn_targets_np = []
    syn_target_masks_np = []

    # Creates list of non-<eos> words
    word_list = word_bank.copy()
    if '<eos>' in word_list:
        word_list.remove('<eos>')
    if '<sos>' in word_list:
        word_list.remove('<sos>')

    # Creates phrases consisting of a single word  
    for word_idx in range(len(word_list)):
        syn_input_np = np.zeros((toy_params['phrase_length'], toy_params['input_size'],))
        syn_target_np = np.zeros((toy_params['phrase_length'], out_size,), dtype=np.int32)
        syn_target_mask_np = np.zeros((toy_params['phrase_length'], out_size,), dtype=np.int32)

        for seq_idx in range(toy_params['phrase_length']):
            syn_input_np[seq_idx] = toy_params['word_to_input_vector'][word_list[word_idx]]
            if word_idx < toy_params['n_classes']:
                syn_target_np[seq_idx, wordToIndex(word_list[word_idx], word_bank)] = 1

        if toy_params['include_eos']:
            syn_input_np[-1] = toy_params['word_to_input_vector']['<eos>']
        if toy_params['include_sos']:
            syn_input_np[0] = toy_params['word_to_input_vector']['<sos>']

        syn_target_mask_np[-1] = np.ones((out_size,), dtype=np.int32)
        
        syn_inputs_np.append(syn_input_np)
        syn_targets_np.append(syn_target_np)
        syn_target_masks_np.append(syn_target_mask_np)

    # Creates phrases consisting of new words not seen by the network
    # Uses toy_params to generate this
    new_word_word_to_input = generateInputVectors(toy_params)
    new_word_count = len(new_word_word_to_input.keys())
    if toy_params['include_eos']:
        new_word_count -= 1
    if toy_params['include_sos']:
        new_word_count -= 1

    for word_idx in range(new_word_count):
        syn_input_np = np.zeros((toy_params['phrase_length'], toy_params['input_size'],))
        syn_target_np = np.zeros((toy_params['phrase_length'], out_size,), dtype=np.int32)
        syn_target_mask_np = np.zeros((toy_params['phrase_length'], out_size,), dtype=np.int32)

        for seq_idx in range(toy_params['phrase_length']):
            syn_input_np[seq_idx] = new_word_word_to_input[toy_params['words'][word_idx]]

        if toy_params['include_eos']:
            syn_input_np[-1] = toy_params['word_to_input_vector']['<eos>']
        if toy_params['include_sos']:
            syn_input_np[0] = toy_params['word_to_input_vector']['<sos>']

        syn_target_mask_np[-1] = np.ones((out_size,), dtype=np.int32)
        
        syn_inputs_np.append(syn_input_np)
        syn_targets_np.append(syn_target_np)
        syn_target_masks_np.append(syn_target_mask_np)

    syn_inputs_np = np.array(syn_inputs_np)
    syn_targets_np = np.array(syn_targets_np, dtype=np.int32)
    syn_target_masks_np = np.array(syn_target_masks_np, dtype=np.int32)

    syn_targets_np = np.argmax(syn_targets_np, axis=-1)[:, :, np.newaxis] # Compresses last dimension to max argument

    # Stacks phrases end to end to create longer temporal streams. Also stacks masks and targets.
    # if toy_params['stack_phrases']:
    #     if toy_params['variable_length']:
    #         raise NotImplementedError()

    #     phrase_len = toy_params['phrase_length']
       
    #     syn_inputs_stack = np.zeros((n_stacked_datasets, phrase_len*toy_params['n_stack'], toy_params['input_size'],))
    #     syn_targets_stack = np.zeros((n_stacked_datasets, phrase_len*toy_params['n_stack'], 1,))
    #     syn_target_masks_stack  = np.zeros((n_stacked_datasets, phrase_len*toy_params['n_stack'], out_size,))

    #     stack_count = 0
    #     batch_count = 0
    #     for data_idx in range(dataset_size):
    #         syn_inputs_stack[batch_count, stack_count*phrase_len:(stack_count+1)*phrase_len, :] = syn_inputs_np[data_idx, :, :]
    #         syn_targets_stack[batch_count, stack_count*phrase_len:(stack_count+1)*phrase_len, :] = syn_targets_np[data_idx, :, :]
    #         syn_target_masks_stack[batch_count, stack_count*phrase_len:(stack_count+1)*phrase_len, :] = syn_target_masks_np[data_idx, :, :]
    #         stack_count += 1
    #         if stack_count == toy_params['n_stack']:
    #             stack_count = 0
    #             batch_count += 1
    #     syn_inputs_np = syn_inputs_stack
    #     syn_targets_np =syn_targets_stack
    #     syn_target_masks_np = syn_target_masks_stack

        # syn_inputs_np = np.reshape(syn_inputs_np, (phrase_len*toy_params['n_stack'], n_stacked_datasets, len(toy_params['words'])))
        # syn_targets_np = np.reshape(syn_targets_np, (phrase_len*toy_params['n_stack'], n_stacked_datasets, 1))
        # syn_target_masks_np = np.reshape(syn_target_masks_np, (phrase_len*toy_params['n_stack'], n_stacked_datasets, out_size))

    # Lastly, converts to pyTorch tensors
    syn_inputs = torch.tensor(syn_inputs_np, dtype=torch.float, device=device)
    syn_targets = torch.tensor(syn_targets_np, dtype=torch.long, device=device)
    syn_target_masks = torch.tensor(syn_target_masks_np, dtype=torch.bool, device=device)

    # print('Input shape:', syn_inputs.shape)
    # print('Target shape:', syn_targets.shape)

    dataset = TensorDataset(syn_inputs, syn_targets)

    return dataset, syn_target_masks


def generate_data_context(dataset_size, toy_params, out_size, auto_balance=False, verbose=True, raw_data=False, device='cpu'):
    """
    Generates contextual integration data for two separate integration tasks

    Essentially just wraps two generate_data calls, and concatanates them together based on a context mask that is randomly generated
    """

    # If needed, does some initialization of the wrapped toy_params_1 (and _2)
    if 'toy_params_1' not in toy_params:
        toy_params_1 = copy.deepcopy(toy_params)
        toy_params_2 = copy.deepcopy(toy_params)

        assert toy_params['input_size'] % 2 == 0

        # Make input size smaller for both networks
        toy_params_1['input_size'] = toy_params['input_size'] // 2
        toy_params_2['input_size'] = toy_params['input_size'] // 2 

        # Now run generate_data real quick to set some default parameters such as 'words' and 'word_to_input_vector'
        _, _, toy_params_1 = generate_data(1, toy_params_1, out_size, auto_balance=False, verbose=False)
        _, _, toy_params_2 = generate_data(1, toy_params_2, out_size, auto_balance=False, verbose=False)

        # Make sure both networks have the same EoS character
        if not toy_params['include_eos']:
            raise NotImplementedError('Havent done this without EoS yet')
        else:
            toy_params_2['word_to_input_vector']['<eos>'] = toy_params_1['word_to_input_vector']['<eos>']

        # # Forces same input vectors
        # toy_params_2['word_to_input_vector'] = toy_params_1['word_to_input_vector']

        toy_params['toy_params_1'] = toy_params_1
        toy_params['toy_params_2'] = toy_params_2

        # Generates random input vectors for the context1 and context2
        # (does this by just taking whatever was randomly generated for evid0 and evid1)
        toy_params['words'] = ['context1', 'context2']
        toy_params['word_to_input_vector'] = generateInputVectors(toy_params)
        # word_to_input_vector = generateInputVectors(toy_params)
        # toy_params['word_to_input_vector'] = {}
        # toy_params['word_to_input_vector']['context1'] = word_to_input_vector['evid0']
        # toy_params['word_to_input_vector']['context2'] = word_to_input_vector['evid1']

    # Now run the true generate_datas
    inputs_1, targets_1, target_masks_1, raw_data_1, toy_params['toy_params_1'] = generate_data(
        dataset_size, toy_params['toy_params_1'], out_size, auto_balance=auto_balance, 
        verbose=False, create_tensors=False, raw_data=True)
    inputs_2, targets_2, target_masks_2, raw_data_2, toy_params['toy_params_2'] = generate_data(
        dataset_size, toy_params['toy_params_2'], out_size, auto_balance=auto_balance, 
        verbose=False, create_tensors=False, raw_data=True)

    # Rescales inputs to appropriate normalization
    if toy_params['input_type'] in ('binary', 'binary1-1'):
        inputs_1 = np.sqrt(toy_params['toy_params_1']['input_size']/toy_params['input_size']) * inputs_1
        inputs_2 = np.sqrt(toy_params['toy_params_2']['input_size']/toy_params['input_size']) * inputs_2
    else:
        raise NotImplementedError()

    # Contextual input, (1, 0) is for first dataset vs. (0, 1) is for second
    # by default context is passed as input for the entire sequence
    context_mask  = np.random.randint(2, size=dataset_size)
    # # Context is 2 one-hots at end of input
    # context_input = np.zeros((inputs_1.shape[0], 1, 2))
    # context_input[:, 0, 0] = context_mask
    # context_input[:, 0, 1] = 1 - context_mask

    # Context is random binary vectors
    context_input = np.where(context_mask[:, np.newaxis], 
                             toy_params['word_to_input_vector']['context1'][np.newaxis, :],
                             toy_params['word_to_input_vector']['context2'][np.newaxis, :])
    context_input = context_input[:, np.newaxis, :] # Create sequence dimension

    # Repeats context across entire sequence
    context_input = np.repeat(context_input, repeats=inputs_1.shape[1], axis=1)
    # Optional modification to context to change the task type
    if toy_params['data_type'] == 'retro_context_int':
        assert abs(toy_params['n_delay']) > 10
        # In this case, context lasts for the 5 time steps before final 5 time steps of phrase
        if toy_params['n_delay'] > 0:
            pre_context = toy_params['phrase_length'] - 10
            post_context = toy_params['phrase_length'] - 5

            epoch_idxs = (
                toy_params['phrase_length'] - toy_params['n_delay'], # Onset of delay
                pre_context, # Onset of context
                post_context, # Onset of post-context
            )
           
            context_input[:, :pre_context, :] = np.zeros((dataset_size, pre_context, context_input.shape[-1]))
            context_input[:, post_context:, :] = np.zeros((dataset_size, toy_params['phrase_length']-post_context, context_input.shape[-1]))
        elif toy_params['n_delay'] < 0:
            pre_context = np.abs(toy_params['n_delay']) - 10
            post_context = np.abs(toy_params['n_delay']) - 5

            epoch_idxs = (
                pre_context, # Onset of context
                post_context, # Onset of post-context
                toy_params['phrase_length'] - toy_params['n_delay'], # End of delay
            )
           
            context_input[:, :pre_context, :] = np.zeros((dataset_size, pre_context, context_input.shape[-1]))
            context_input[:, post_context:, :] = np.zeros((dataset_size, toy_params['phrase_length']-post_context, context_input.shape[-1]))

    # Filters targets by context
    targets = np.zeros((dataset_size, toy_params['phrase_length'], 1), dtype=np.int32)
    targets[:, -1, 0] = np.where(context_mask, targets_1[:, -1, 0], targets_2[:, -1, 0])

    # Now put the input data together
    # raw_inputs = np.concatenate((
    #         inputs_1, inputs_2, context_input
    #     ), axis=-1)
    raw_inputs = np.concatenate((inputs_1, inputs_2,), axis=-1)
    raw_inputs = raw_inputs + context_input

    # Lastly, converts to pyTorch tensors
    syn_inputs = torch.tensor(raw_inputs, dtype=torch.float, device=device)
    syn_targets = torch.tensor(targets, dtype=torch.long, device=device)
    syn_target_masks = torch.tensor(target_masks_1, dtype=torch.bool, device=device)

    dataset = TensorDataset(syn_inputs, syn_targets)


    if raw_data:
        phrases_1 = raw_data_1[0]
        phrases_2 = raw_data_2[0]
        raw_data = (context_mask, phrases_1, phrases_2, targets_1, targets_2, epoch_idxs)
        return dataset, syn_target_masks, raw_data, toy_params
    else:
        return dataset, syn_target_masks, toy_params

def generate_data_anti_context(dataset_size, toy_params, out_size, auto_balance=False, verbose=True, raw_data=False, device='cpu'):
    """
    Generates contextual integration data for the anti-task
    """

    # If needed, does some initialization of the wrapped toy_params_1 (and _2)
    assert toy_params['n_classes'] == 2 # This could work for more classes, but not yet implemented

    # Not really necessary, but uses the same setup of 2-task context integration to avoid some uniform score dist stuff
    if 'toy_params_1' not in toy_params:
        toy_params_1 = copy.deepcopy(toy_params)

        # Now run generate_data real quick to set some default parameters such as 'words' and 'word_to_input_vector'
        _, _, toy_params_1 = generate_data(1, toy_params_1, out_size, auto_balance=False, verbose=False)

        toy_params['toy_params_1'] = toy_params_1

        # Generates random input vectors for the context1 and context2
        # (does this by just taking whatever was randomly generated for evid0 and evid1)
        toy_params['words'] = ['context1', 'context2']
        toy_params['word_to_input_vector'] = generateInputVectors(toy_params)

    # Now run the true generate_data
    inputs_1, targets_1, target_masks_1, toy_params['toy_params_1'] = generate_data(
        dataset_size, toy_params['toy_params_1'], out_size, auto_balance=False, verbose=False, create_tensors=False)
    anti_targets_1 = (targets_1 + 1) % 2

    # Contextual input, 1 is for normal targets vs. 0 is for anti-targets
    # by default context is passed as input for the entire sequence
    context_mask  = np.random.randint(2, size=dataset_size)

    # Context is random binary vectors
    context_input = np.where(context_mask[:, np.newaxis], 
                             toy_params['word_to_input_vector']['context1'][np.newaxis, :],
                             toy_params['word_to_input_vector']['context2'][np.newaxis, :])
    context_input = context_input[:, np.newaxis, :] # Create sequence dimension

    # Repeats context across entire sequence
    context_input = np.repeat(context_input, repeats=inputs_1.shape[1], axis=1)
    # Optional modification to context to change the task type
    if toy_params['data_type'] == 'retro_context_anti':
        assert abs(toy_params['n_delay']) > 10
        # In this case, context lasts for the 5 time steps before final 5 time steps of phrase
        if toy_params['n_delay'] > 0:
            pre_context = toy_params['phrase_length'] - 10
            post_context = toy_params['phrase_length'] - 5

            epoch_idxs = (
                toy_params['phrase_length'] - toy_params['n_delay'], # Onset of delay
                pre_context, # Onset of context
                post_context, # Onset of post-context
            )
           
            context_input[:, :pre_context, :] = np.zeros((dataset_size, pre_context, context_input.shape[-1]))
            context_input[:, post_context:, :] = np.zeros((dataset_size, toy_params['phrase_length']-post_context, context_input.shape[-1]))
        elif toy_params['n_delay'] < 0:
            pre_context = np.abs(toy_params['n_delay']) - 10
            post_context = np.abs(toy_params['n_delay']) - 5

            epoch_idxs = (
                pre_context, # Onset of context
                post_context, # Onset of post-context
                toy_params['phrase_length'] - np.abs(toy_params['n_delay']), # End of delay
            )
           
            context_input[:, :pre_context, :] = np.zeros((dataset_size, pre_context, context_input.shape[-1]))
            context_input[:, post_context:, :] = np.zeros((dataset_size, toy_params['phrase_length']-post_context, context_input.shape[-1]))

    # Filters targets by context
    targets = np.zeros((dataset_size, toy_params['phrase_length'], 1), dtype=np.int32)
    targets[:, -1, 0] = np.where(context_mask, targets_1[:, -1, 0], anti_targets_1[:, -1, 0])

    # Now put the input data together
    # raw_inputs = np.concatenate((
    #         inputs_1, inputs_2, context_input
    #     ), axis=-1)
    raw_inputs = inputs_1 + context_input

    # Lastly, converts to pyTorch tensors
    syn_inputs = torch.tensor(raw_inputs, dtype=torch.float, device=device)
    syn_targets = torch.tensor(targets, dtype=torch.long, device=device)
    syn_target_masks = torch.tensor(target_masks_1, dtype=torch.bool, device=device)

    dataset = TensorDataset(syn_inputs, syn_targets)

    if raw_data:
        raw_data = (context_mask, targets_1, anti_targets_1,)
        return dataset, syn_target_masks, raw_data, toy_params
    else:
        return dataset, syn_target_masks, toy_params

#######################################################################################
############### Functions having to do with uniform score distributions ############### 
#######################################################################################

def enumerate_phrases(toy_params):
    """ 
    Enumerates all possible phrases.
    Does this by creating a one-to-one correspondence of all possible phrases
    and a corresponding index from 0 to num of possible phrases.
    Works for all possible word banks but can be VERY SLOW.
    
    OUTPUTS:
    score_dict: dictionary with keys of possible scores (multi-dim) and value of list of possible phrase indexes
    """
    
    phrase_length = toy_params['phrase_length']
    words = toy_params['words']

    total_phrases = len(words)**phrase_length

    score_dict = {}
    scores = []
    for phrase_idx in range(total_phrases):
        phrase = index_to_phrase(phrase_idx, toy_params)
        score = score_toy_phrase(phrase, toy_params)
        if score in score_dict:
            score_dict[score].append(phrase_idx)
        else:
            score_dict[score] = [phrase_idx]

    return score_dict

def index_to_phrase(index, toy_params):
    """ 
    Converts an index to a phrase 

    Essentially just associates each phrase to n-nary number, 
    with n the possible number of words, and converts
    it to base ten 

    """
    phrase_length = toy_params['phrase_length']
    words = toy_params['words']

    phrase = ['' for _ in range(phrase_length)]
    for idx in range(phrase_length):
        index, rem = divmod(index, len(words))
        phrase[idx] = words[rem]
      
    return phrase

def score_to_phrase_unordered(score, phrase_length, toy_params):
    """ 
    An quicker way of generating phrases for uniform scores that doesn't require
    enumerating all possible phrases, but so far only works for certain word collections 

    """
    # print('Phrase length in score_to_phrase', phrase_length)
    words = toy_params['words']

    phrase = []

    for score_idx in range(len(score)):
        phrase.extend(['evid'+str(score_idx) for _ in range(int(score[score_idx]))])
    phrase.extend(['null' for _ in range(phrase_length - len(phrase))])

    np.random.shuffle(phrase)
      
    return phrase

def enumerate_scores(toy_params):
    """ Converts a score to a phrase, only works for certain word collections """
    
    phrase_length = toy_params['phrase_length']
    n_unordered = toy_params['n_unordered']
    # filter = toy_params['filter'] if 'filter' in toy_params else 0

    scores = []
    if n_unordered == 2:
        for score1 in range(phrase_length+1):
            scores.append([score1, phrase_length - score1])
        scores_filtered = scores
    if n_unordered == 3:
        for score1 in range(phrase_length+1):
            for score2 in range(phrase_length - score1+1):
                scores.append([score1, score2, phrase_length - score1 - score2])
        scores_filtered = []
        for score_idx in range(len(scores)):
            score = scores[score_idx]
            if score[0] <= (phrase_length/2 - filter) or score[1] <= (phrase_length/2 - filter):
                scores_filtered.append(score)
    else:
        raise NotImplementedError()

    return scores_filtered

def check_word_bank_for_uniform_score(words, n_classes):
    """ 
    Checks if current word bank can be used for uniform score generation.
    Assumes scores are of the form {'evid0', 'evid1', ..., 'evid(n_classes-1)', 'null'}
    """

    words_to_check = words.copy()

    # Special characters are not included in the check
    if '<eos>' in words:
        words_to_check.remove('<eos>')
    if '<sos>' in words:
        words_to_check.remove('<sos>')
    if '<delay>' in words:
        words_to_check.remove('<delay>')

    contradiction = False
    for i in range(n_classes):
        if 'evid'+str(i) not in words_to_check:
            contradiction = True
    if 'null' not in words_to_check:
        contradiction = True
    if len(words_to_check) != n_classes+1:
        contradiction = True
    if contradiction:
        raise ValueError('Word bank incompatible with uniform scores.')

def n_scores(n_classes, phrase_length):
    """ 
    Determines the number of possible scores for n_classes and a phrase length of max.
    Assumes scores are of the form {'evid0', 'evid1', ..., 'evid(n_classes-1)', 'null'}
    
    """
    return int(
        np.prod(np.array([n + 1 + phrase_length for n in range(0, n_classes)], dtype=np.float128)) / 
        np.prod(np.array([n + 1 for n in range(n_classes)], dtype=np.float128))
    )

def index_to_score_unordered(index, n_classes, phrase_length):
    """ 
    Map an index to a score, used to generate scores with uniform probability.
    Assumes scores are of the form {'evid0', 'evid1', ..., 'evid(n_classes-1)', 'null'}
    """
    
    if index >= n_scores(n_classes, phrase_length):
        raise ValueError('Index is too large!')
    score = np.zeros((n_classes,))
    count = phrase_length.copy()
    n = n_classes - 1
    for score_idx in range(n_classes):
        if score_idx == n_classes - 1:
            score[score_idx] = index
        else:
            current_val = 0
            val_found = False
            while not val_found:
                if index < n_scores(n, count):
                    val_found = True
                    score[score_idx] = current_val
                    n -= 1
                else:
                    index -= n_scores(n, count)
                    count -= 1
                    current_val += 1

    return score

def check_word_bank_for_uniform_score_ordered(words):
    """ 
    Checks if current word bank can be used for uniform score generation ordered.
    Assumes scores are of the form {'good', 'bad', 'the'}
    """
    contradiction = False
    
    if len(words) == 3:
        if 'good' not in words:
            contradiction = True
        if 'bad' not in words:
            contradiction = True
        if 'the' not in words:
            contradiction = True
    elif len(words) == 5:
        if words != ['awesome', 'good', 'bad', 'awful', 'the']: # Needs this exact ordering for now
            contradiction = True
    elif len(words) == 6:
        if words != ['awesome', 'good', 'bad', 'awful', 'okay', 'the']: # Needs this exact ordering for now
            contradiction = True
    else:
        contradiction = True
    if contradiction:
        raise ValueError('Word bank incompatible with uniform scores.')

def index_to_score_ordered(index, max_val):
    """ 
    Map an index to a score, used to generate scores with uniform probability for ordered classes.
    """
    if index >= (2*max_val+1):
        raise ValueError('Index is too large!')
    return index - max_val

def score_to_phrase_ordered(score, toy_params):
    """ Converts a score to a phrase, only works for certain word collections """
    phrase_length = toy_params['phrase_length']

    phrase = []
    if score > 0:
        phrase.extend(['good' for _ in range(int(score))])
    elif score < 0:
        phrase.extend(['bad' for _ in range(int(np.abs(score)))])
    if score % 2 == 1: # Odd scores get a 'the'
        phrase.extend(['the'])

    # Randomly determines the number of 'the' or 'good/bad' occurences
    max_zero_sums = int((phrase_length - np.abs(score)) // 2)
    n_the_pairs = np.random.randint(max_zero_sums+1)
    
    for _ in range(n_the_pairs):
        phrase.extend(['the', 'the'])
    for _ in range(max_zero_sums-n_the_pairs):
        phrase.extend(['good', 'bad'])

    np.random.shuffle(phrase)
      
    return phrase

def generate_score_map(n_classes, length, score_vals):
    """ Used fo uniform score distribution of ordered datasets with more words than just good, bad, the """
    print('Generating score map for uniform scores...')
    total_scores = int(n_scores(n_classes, length))
    score_map = {}
    for score_idx in range(total_scores):
        raw_count = index_to_score_unordered(score_idx, n_classes, length)
        # print('Raw count', raw_count)
        # print('score vals', score_vals)
        score = tuple(np.dot(raw_count, score_vals))
        if score in score_map:
            score_map[score].append(score_idx)
        else:
            score_map[score] = [score_idx]
    return score_map

def score_to_phrase_ordered_general(score, score_map, toy_params):
    """ Converts a score to a phrase, only works for certain word collections """
    phrase_length = toy_params['phrase_length']
    words = toy_params['words']

    score = tuple(score) # converts the score to a for indexing score_map
    phrase = []
    # Random index for the given score
    phrase_idx = np.random.randint(len(score_map[score]))
    # Raw count of words that are not 'the'
    raw_count = index_to_score_unordered(score_map[score][phrase_idx], len(words)-1, phrase_length)

    for count_idx in range(len(raw_count)):
        phrase.extend([words[count_idx] for _ in range(int(raw_count[count_idx]))])
    phrase.extend(['the' for _ in range(phrase_length - len(phrase))])

    np.random.shuffle(phrase)
      
    return phrase
