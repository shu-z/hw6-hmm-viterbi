import copy
import numpy as np
class ViterbiAlgorithm:
    """_summary_
    """    

    def __init__(self, hmm_object):
        """_summary_

        Args:
            hmm_object (_type_): _description_
        """              
        self.hmm_object = hmm_object

    def best_hidden_state_sequence(self, decode_observation_states: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            decode_observation_states (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """        
        
       # Initialize path (i.e., np.arrays) to store the hidden sequence states returning the maximum probability
        path = np.zeros((len(decode_observation_states), 
                         len(self.hmm_object.hidden_states)))
        path[0,:] = [hidden_state_index for hidden_state_index in range(len(self.hmm_object.hidden_states))]

        best_path = np.zeros(len(decode_observation_states))        
        
        
        print(self.hmm_object.hidden_states_dict)
        trans_p=self.hmm_object.transition_probabilities
        emis_p=self.hmm_object.emission_probabilities
        

        init_delta = np.multiply(self.hmm_object.prior_probabilities, self.hmm_object.emission_probabilities.T)
        obs_idx=self.hmm_object.observation_states_dict[decode_observation_states[0]]
        delta=init_delta[obs_idx,:]

        # For each observation state to decode, select the hidden state sequence with the highest probability (i.e., Viterbi trellis)
        for trellis_node in range(1, len(decode_observation_states)):
            
            
            print('node id: ' + str(trellis_node))
    
            
            #get integer value representing current observed state            
            obs_idx=self.hmm_object.observation_states_dict[decode_observation_states[trellis_node]]

            
            product_of_delta_and_trans_emis = np.multiply(delta, trans_p.T)
            product_of_delta_and_trans_emis=np.multiply(product_of_delta_and_trans_emis.T, emis_p[:, obs_idx])
            
            
           # print('product delta transition emission')
            #print(product_of_delta_and_trans_emis)
            
            
           # print('old_delta', delta)
        
            old_delt_list=np.zeros(len(self.hmm_object.hidden_states))
            
            for hidden_state in range(len(self.hmm_object.hidden_states)):
                #print('product', product_of_delta_and_trans_emis[:,hidden_state])
                #print(np.argmax(product_of_delta_and_trans_emis[:,hidden_state]))
                
                
                test1=np.max(product_of_delta_and_trans_emis[:,hidden_state])
                old_delt_list[hidden_state]=test1
                path[trellis_node, hidden_state]=np.argmax(product_of_delta_and_trans_emis[:,hidden_state])
    
                
                
                delta[hidden_state]=np.max(product_of_delta_and_trans_emis[:,hidden_state])
            

            test1=np.argmax(product_of_delta_and_trans_emis)
            best_path[trellis_node-1] = np.argmax(old_delt_list)
   

        #backtrace through best_path
        for node in reversed(range(1,len(decode_observation_states))):
            best_path[node-1] = path[node, int(best_path[node])]
            
        
        hidden_dict=self.hmm_object.hidden_states_dict
        best_hidden_state_path = [hidden_dict[i] for i in best_path]

        return best_hidden_state_path