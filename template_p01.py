import numpy as np

def softmax(vector):
    '''
    vector: np.array of shape (n, m)
    
    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    '''
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    dot_product = np.dot(decoder_hidden_state, encoder_hidden_states) / np.sqrt(W_mult)
    
    probabilities = np.exp(dot_product) / np.sum(np.exp(dot_product), axis=1, keepdims=True)
    
    attention_vector = np.sum(value * probabilities[:,:,None], axis=1)

    return attention_vector

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    # Compute the dot product of the query and key
    dot_product = np.dot(decoder_hidden_state, encoder_hidden_states)
    
    # Apply the tanh function to the dot product tensor
    tanh_output = np.tanh(dot_product)
    
    # Compute the dot product of the tanh output and the attention weights
    attention_scores = np.dot(tanh_output, W_add_enc) + np.dot(query, W_add_dec)
    
    # Apply the softmax function to the attention scores
    probabilities = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)
    
    # Compute the weighted sum of the value tensors using the probabilities
    # as the weights
    attention_vector = np.sum(v_add * probabilities[:,:,None], axis=1)
    
    return attention_vector
