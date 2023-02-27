import copy
import numpy as np
class ViterbiAlgorithm:
    """ Class to run the Viterbi Algorithm from an HMM object. 
    """    

    def __init__(self, hmm_object):
        """ Initializes hmm object

        Args:
            hmm_object (_type_): Object containing observed states, hidden states, and probabilities for running the Viterbi Algorithm
        """              
        self.hmm_object = hmm_object

    def best_hidden_state_sequence(self, decode_observation_states: np.ndarray) -> np.ndarray:
        """Calculates best sequences of hidden states from observed states

        Args:
            decode_observation_states (np.ndarray): Sequence of observation states, to be decoded into corresponding sequence of hidden states

        Returns:
            np.ndarray: Array with best hidden state sequence as determined by the Viterbi Algorithm
        """ 

        #init relevant variables
        trans_p=self.hmm_object.transition_probabilities
        emis_p=self.hmm_object.emission_probabilities
        obs_dict=self.hmm_object.observation_states_dict
        hidden_dict=self.hmm_object.hidden_states_dict
        hidden_s=self.hmm_object.hidden_states

        
       # Initialize path (i.e., np.arrays) to store the hidden sequence states returning the maximum probability
        path = np.zeros((len(decode_observation_states), len(hidden_s)))
        path[0,:] = [hidden_state_index for hidden_state_index in range(len(hidden_s))]
        best_path = np.zeros(len(decode_observation_states))        
        
        
        #init delta, a product of prior and emissions  probabilities
        init_delta = np.multiply(self.hmm_object.prior_probabilities, emis_p.T)
        obs_idx=obs_dict[decode_observation_states[0]]
        #delta is based on index 0 observation state
        delta=init_delta[obs_idx,:]


        # For each observation state to decode, select the hidden state sequence with the highest probability (i.e., Viterbi trellis)
        for trellis_node in range(1, len(decode_observation_states)):

            #get integer value representing current observation state            
            obs_idx=obs_dict[decode_observation_states[trellis_node]]
         
            #get probabilities for paths going to next state
            #product of delta, transition, emission 
            product_of_delta_and_trans_emis = np.multiply(delta, trans_p.T)
            product_of_delta_and_trans_emis=np.multiply(product_of_delta_and_trans_emis.T, emis_p[:, obs_idx])



            #keep track of previous delta to update path later
            old_delt_list=np.zeros(len(hidden_s))
            
            #loop through each hidden state
            for hidden_state in range(len(hidden_s)):

                #get maximum path probabilities for the hidden state column
                prod_max_hidden_state=np.max(product_of_delta_and_trans_emis[:,hidden_state])
                old_delt_list[hidden_state]=prod_max_hidden_state
                #update path 
                path[trellis_node, hidden_state]=np.argmax(product_of_delta_and_trans_emis[:,hidden_state])
    
                #update delta (must do after updating path)
                delta[hidden_state]=prod_max_hidden_state
            

            best_path[trellis_node-1] = np.argmax(old_delt_list)
   

        #backtrace through best_path
        for node in reversed(range(1,len(decode_observation_states))):
            best_path[node-1] = path[node, int(best_path[node])]
               
        #turn indices into sequence of hidden state names
        best_hidden_state_path = [hidden_dict[i] for i in best_path]

        return best_hidden_state_path