import numpy as np
from collections import namedtuple
import pandas as pd
import dill
from copy import copy

from .ddpg_event_graph_offline import DDPG_Event_Graph_Off


class DDPG_Event_Graph_Off_NL(DDPG_Event_Graph_Off):
    def __init__(self, config, agent_config, is_eval) -> None:
        super(DDPG_Event_Graph_Off_NL, self).__init__(
            config, agent_config, is_eval)

        self.__behav_policy = agent_config['behav_policy']
        self.__pertu_range = agent_config['pertu_range']
        self.__num_day = agent_config['requi_num_day']

        if not self._is_eval:
            self.form_memory()
            self.offline_learn()
            self.save_model(self._actor_net.state_dict(), agent_config)
        else:
            model = self.load_model(agent_config)
            self._actor_net.load_state_dict(model)

    def form_memory(self):
        # form transition file name
        file = 'data/' + self.__behav_policy
        file += '_day_' + str(self.__num_day)
        file += '_sg_' + str(self._is_state_globa)
        file += '_rg_' + str(self._is_rewar_globa)
        file += '_p_' + str(self.__pertu_range)
        file += '_eg_trans.pickle'
        self.__tran_file = file
        print(self.__tran_file)

        with open(self.__tran_file, 'rb') as f:
            while True:
                try:
                    tran = dill.load(f)
                    _, _, _, _, s, a, equal_r, n_s, g, n_g = tran
                    r = equal_r - self._w * a
                    assert a >= 0 and a <= 1, 'action should be in [0, 1]'
                    assert equal_r <= 0 and r <= 0, 'reward should be negative'
                    self._memor.append([s, a, r, n_s, g, n_g])
                except EOFError:
                    break

    def __str__(self) -> str:
        return 'DDPG_EG_OFF_NONLINEAR'

    # def reset(self, episode, is_record_wandb=False, is_record_transition=False):
    #     # write to wandb
    #     super().reset(episode, is_record_wandb)
    #     self.__event_handl.clear_events()

    # def cal_hold_time(self, snapshot):
    #     if self._is_state_globa:
    #         actio = self.infer(snapshot.globa_relat_state)
    #     else:
    #         actio = self.infer(snapshot.local_state)
    #     hold_time = actio.item() * self._max_hold
    #     # record departure event and log reward
    #     track_equal_rewar, track_inten_rewar, track_rewar = self.__event_handl.add_event(
    #         snapshot, actio, hold_time, self._w)
    #     self.track(track_rewar, track_equal_rewar,
    #                track_inten_rewar, hold_time)
    #     return hold_time

    # def infer(self, state):
    #     with torch.no_grad():
    #         a = self.__actor_net(state)
    #     return a
