"""
UCSF BMI203: Biocomputing Algorithms
Author:
Date: 
Program: 
Description:
"""
import pytest
import numpy as np
from src.models.hmm import HiddenMarkovModel
from src.models.decoders import ViterbiAlgorithm


def check_hmm_dims(viterbi_obj, hidden_states, obs_states):

    """Checks that correct dimensions exist for prior, transition, emission probability matrices
    """


    #check that prior probability dims are equal to # of hidden states
    assert np.allclose(len(viterbi_obj.hmm_object.prior_probabilities), len(hidden_states))

    #get shapes of transition and emission matrices
    transition_shape=viterbi_obj.hmm_object.transition_probabilities.shape
    emission_shape=viterbi_obj.hmm_object.emission_probabilities.shape

    #check that transition prob dim is nxn where n is # hidden states
    assert np.allclose(transition_shape[0], transition_shape[1])
    assert np.allclose(transition_shape[0], len(hidden_states))

    #check that emission prob dim is nxm where n is # hidden states and m is # observed states 
    assert np.allclose(emission_shape[0], len(hidden_states))
    assert np.allclose(emission_shape[1], len(obs_states))




def test_use_case_lecture():
    """Tests hypothesis that a rotation student's dedication is dependent on their lab's NIH funding. 
    Checks that correct hidden sequence is calculated, and variable dimensions are as expected.  
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['committed','ambivalent'] # A graduate student's dedication to their rotation lab
    
    # index annotation hidden_states=[i,j]
    hidden_states = ['R01','R21'] # The NIH funding source of the graduate student's rotation project 

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-Lecture.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])


     # TODO: Check HMM dimensions and ViterbiAlgorithm

    #check a bunch of matrix dimensions
    check_hmm_dims(use_case_one_viterbi, hidden_states, observation_states)

    #check that length of best path (viterbi output) is same length as input observation states
    assert len(use_case_decoded_hidden_states)==len(use_case_one_data['observation_states'])





def test_user_case_one():
    """Tests hypothesis that an individual's timeliness is dependent on the presence of traffic. 
    Checks that correct hidden sequence is calculated, and variable dimensions are as expected.  
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['on-time','late'] 

    # index annotation hidden_states=[i,j]
    hidden_states = ['no-traffic','traffic']

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-One.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])


    # TODO: Check HMM dimensions and ViterbiAlgorithm

    #check a bunch of matrix dimensions
    check_hmm_dims(use_case_one_viterbi, hidden_states, observation_states)

    #check that length of best path (viterbi output) is same length as input observation states
    assert len(use_case_decoded_hidden_states)==len(use_case_one_data['observation_states'])



def test_user_case_two():
    """Tests hypothesis that the outside temperature is dependent on whether it is sunny or rainy. 
    Checks that correct hidden sequence is calculated, and variable dimensions are as expected.  
    """
    # TODO

    use_case_two_data = np.load('./data/UserCase-Two.npz')
    observation_states=['Hot', 'Cold']
    hidden_states=['Sun', 'Rain']

    use_case_two_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_two_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_two_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_two_data['emission_probabilities']) 

     # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_two_viterbi = ViterbiAlgorithm(use_case_two_hmm)


      # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_two_viterbi.best_hidden_state_sequence(use_case_two_data['observation_states'])
    print(use_case_decoded_hidden_states)
    print(use_case_two_data['hidden_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_two_data['hidden_states'])


    # TODO: Check HMM dimensions and ViterbiAlgorithm

   
    #check a bunch of matrix dimensions
    check_hmm_dims(use_case_two_viterbi, hidden_states, observation_states)

    #check that length of best path (viterbi output) is same length as input observation states
    assert len(use_case_decoded_hidden_states)==len(use_case_two_data['observation_states'])





    


def test_user_case_three():
    """Tests hypothesis that an individual's mood is dependent on whether or not they ate lunch. 
    Checks that correct hidden sequence is calculated, and variable dimensions are as expected. 
    Also tests if algorithm still works with different dimensions for observation states and related variables.
    """


    use_case_three_data = np.load('./data/UserCase-Three.npz')
    observation_states=['Happy', 'Ambivalent', 'Grumpy']
    hidden_states=['Ate Lunch', 'No Food']

    use_case_three_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_three_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_three_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_three_data['emission_probabilities']) 

     # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_three_viterbi = ViterbiAlgorithm(use_case_three_hmm)


      # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_three_viterbi.best_hidden_state_sequence(use_case_three_data['observation_states'])
    print(use_case_decoded_hidden_states)
    print(use_case_three_data['hidden_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_three_data['hidden_states'])



    #check a bunch of matrix dimensions
    check_hmm_dims(use_case_three_viterbi, hidden_states, observation_states)

    #check that length of best path (viterbi output) is same length as input observation states
    assert len(use_case_decoded_hidden_states)==len(use_case_three_data['observation_states'])




    
