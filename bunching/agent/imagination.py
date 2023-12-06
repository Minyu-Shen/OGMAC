import numpy as np
from .agent import Agent
import wandb
from .event import Event_Handler
from itertools import product


class Imagination(Agent):
    def __init__(self, config, agent_config, is_eval):
        super(Imagination, self).__init__(config, agent_config, is_eval)
        self.__is_graph = agent_config['is_graph']
        self.__type_dict = {'behav_polic': 'DO_NOTHING', 'is_state_globa': self._is_state_globa,
                            'is_rewar_globa': self._is_rewar_globa, 'is_graph': self.__is_graph}
        self.__type_dict = {'behav_polic': 'IMAGINATION', 'is_state_globa': self._is_state_globa,
                            'is_rewar_globa': self._is_rewar_globa, 'is_graph': self.__is_graph}
        self.__event_handl = Event_Handler(config, is_rewar_globa=self._is_rewar_globa, is_rewar_track_globa=True,
                                           is_state_globa=self._is_state_globa, max_hold=self._max_hold, w=self._w, is_graph=self.__is_graph)

    def __str__(self) -> str:
        return 'IMAGINATION'

    def reset(self, episode, is_record_wandb=False, is_record_transition=False):
        super().reset(episode, is_record_wandb)
        # at the final, write it to csv
        if is_record_transition:
            self.__event_handl.write_transition_to_file(self.__type_dict)

    # note that for imagination, the hold time is not hold time calculated by agent, but the tt-mean_tt in advance
    def get_hold_time(self, mean_tt, sample_travel_time_in_advance, snapshot):
        ct = snapshot.ct
        assert ct == snapshot.last_depar_times[-1]
        hold_time = max(0, sample_travel_time_in_advance - mean_tt)
        hold_time = min(self._max_hold, hold_time)

        # record departure event and log reward
        actio = hold_time / self._max_hold
        track_equal_rewar, track_inten_rewar, track_rewar = self.__event_handl.add_event(
            snapshot, actio, hold_time, self._w)
        self.track(track_rewar, track_equal_rewar,
                   track_inten_rewar, hold_time)

        return hold_time
