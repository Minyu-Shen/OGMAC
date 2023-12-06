import itertools
import numpy as np


class Agent_Config(object):
    def __init__(self, agent_name):
        # some common parameters
        self._w = [0.005]
        self._max_hold = [60.0]
        self._gamma = [0.9]
        self.__agent_config = {'w': self._w,
                               'max_hold': self._max_hold, 'gamma': self._gamma}

        if agent_name == 'DO_NOTHING':
            self.__set_Do_Nothing_sweep()
        elif agent_name == 'NONLINEAR':
            self.__set_Nonlinear_sweep()
        elif agent_name == 'DDPG_EG_ON':
            self.__set_DDPG_EG_On_sweep()
        elif agent_name == 'DDPG_EG_OFF_DO_NOTHING':
            self.__set_DDPG_EG_Off_Do_Nothing_sweep()
        elif agent_name == 'DDPG_EG_OFF_NONLINEAR':
            self.__set_DDPG_EG_Off_Nonlinear_sweep()
        elif agent_name == 'IMAGINATION':
            self.__set_Imagination_sweep()

        self.__sweep = self.__create_sweep()

    def __set_Imagination_sweep(self):
        self.__agent_config['is_state_globa'] = [False]
        self.__agent_config['is_rewar_globa'] = [False]
        self.__agent_config['is_graph'] = [False]
        self.__agent_config['agent_name'] = ['IMAGINATION']

    def __set_Do_Nothing_sweep(self):
        self.__agent_config['is_state_globa'] = [False]
        self.__agent_config['is_rewar_globa'] = [True]
        self.__agent_config['is_graph'] = [True]
        self.__agent_config['agent_name'] = ['DO_NOTHING']

    def __set_Nonlinear_sweep(self):
        self.__agent_config['is_state_globa'] = [False]
        self.__agent_config['is_rewar_globa'] = [True]
        self.__agent_config['is_graph'] = [True]
        self.__agent_config['pertu_range'] = [0]
        self.__agent_config['alpha'] = [0.6]
        self.__agent_config['slack'] = [0]

        self.__agent_config['behav_polic'] = ['NONLINEAR_FIX_UPDATE']
        # self.__agent_config['behav_polic'] = ['NONLINEAR_UPDATE']
        self.__agent_config['agent_name'] = ['NONLINEAR']

    def __set_DDPG_EG_On_sweep(self):
        self.__set_RL_agent_common_config()
        self.__agent_config['is_graph'] = [True]
        self.__agent_config['init_noise_level'] = [0.2]
        self.__agent_config['decay_rate'] = [0.9]
        self.__agent_config['gat_hidde_size'] = [16]
        self.__agent_config['is_rewar_globa'] = [True]
        self.__agent_config['agent_name'] = ['DDPG_EG_ON']

    def __set_DDPG_EG_Off_Do_Nothing_sweep(self):
        self.__set_RL_agent_common_config()
        self.__agent_config['is_graph'] = [True]
        self.__agent_config['gat_hidde_size'] = [16]
        self.__agent_config['eta'] = [0.5]
        self.__agent_config['behav_policy'] = ['DO_NOTHING']
        self.__agent_config['is_rewar_globa'] = [True]
        self.__agent_config['agent_name'] = ['DDPG_EG_OFF_DO_NOTHING']
        # self.__agent_config['requi_num_day'] = [
        #     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        self.__agent_config['requi_num_day'] = [100]

    def __set_DDPG_EG_Off_Nonlinear_sweep(self):
        self.__set_RL_agent_common_config()
        self.__agent_config['is_graph'] = [True]
        self.__agent_config['gat_hidde_size'] = [16]
        self.__agent_config['pertu_range'] = [0]
        self.__agent_config['behav_policy'] = ['NONLINEAR_FIX']
        self.__agent_config['is_rewar_globa'] = [True]
        self.__agent_config['agent_name'] = ['DDPG_EG_OFF_NONLINEAR']
        self.__agent_config['requi_num_day'] = [100]

    def __set_RL_agent_common_config(self):
        self.__agent_config['layer_init_type'] = ['default']
        self.__agent_config['is_state_globa'] = [False]
        self.__agent_config['polya'] = [0.99]
        self.__agent_config['lr'] = [5e-3]
        self.__agent_config['batch_size'] = [64]
        self.__agent_config['hidde_size'] = [[64,]]
        self.__agent_config['is_embed_discr_state'] = [True]

    def __create_sweep(self):
        combinations = list(itertools.product(*self.__agent_config.values()))
        sweep = []
        for combo in combinations:
            result = {}
            for i, key in enumerate(self.__agent_config.keys()):
                result[key] = combo[i]
            sweep.append(result)
        return sweep

    def get_sweep(self):
        return self.__sweep
