import numpy as np
from collections import namedtuple
import pandas as pd
import dill
import random

from .ddpg_event_graph_offline import DDPG_Event_Graph_Off


class DDPG_Event_Graph_Off_DN(DDPG_Event_Graph_Off):
    def __init__(self, config, agent_config, is_eval) -> None:
        super(DDPG_Event_Graph_Off_DN, self).__init__(
            config, agent_config, is_eval)
        self.__behav_policy = agent_config['behav_policy']
        self.__eta = agent_config['eta']
        self.__num_day = agent_config['requi_num_day']
        assert self.__behav_policy == 'DO_NOTHING', 'behav_policy must be DO_NOTHING'
        self.__node_feat = namedtuple(
            'node_feat', ['up_or_down', 'augme_info', 'state', 'action'])

        if not self._is_eval:
            self.form_memory()
            porti = self.__num_day / 100
            random.shuffle(self._memor)
            self._memor = self._memor[:int(len(self._memor)*porti)]
            print('memory size is', len(self._memor))
            self.offline_learn()
            self.save_model(self._actor_net.state_dict(), agent_config)
        else:
            model = self.load_model(agent_config)
            self._actor_net.load_state_dict(model)

    def form_memory(self):
        # form transition file name
        file = 'data/' + self.__behav_policy

        # file += '_day_' + str(self.__num_day)
        file += '_day_' + str(100)

        file += '_sg_' + str(self._is_state_globa)
        file += '_rg_' + str(self._is_rewar_globa)
        # if 'NONLINEAR' in self.__behav_policy:
        #     file += '_p_' + str(self.__pertu_range)
        file += '_eg_trans.pickle'
        self.__tran_file = file
        print(self.__tran_file)

        # mean_tts = [
        #     x / self._config.mean_speed for x in self._config.link_lengs]

        mean_tts = [info['lengt'] / info['mean_speed']
                    for _, info in self._config.link_info.items()]

        tt_header = ['ep', 'bus_id', 'stop_id', 'ct', 'tt']

        # tt_file = 'data/DO_NOTHING_day_' + \
        #     str(self.__num_day) + '_travel_time_eg.csv'
        tt_file = 'data/DO_NOTHING_day_' + \
            str(100) + '_travel_time_eg.csv'

        tt_df = pd.read_csv(tt_file, names=tt_header)

        selec_colus = ['ep', 'bus_id', 'stop_id', 'ct']
        value_colum = 'tt'
        tt_query_dict = dict(zip(zip(tt_df[selec_colus[0]], tt_df[selec_colus[1]],
                                     tt_df[selec_colus[2]], tt_df[selec_colus[3]]), tt_df[value_colum]))

        tt_df['dev'] = tt_df.apply(lambda row: (
            row.tt - mean_tts[int(row.stop_id)])**2, axis=1)
        mean_sigma = np.sqrt(tt_df['dev'].mean())
        ampli = self._max_hold / (self.__eta*mean_sigma)
        print(ampli, self._max_hold, self._w)

        with open(self.__tran_file, 'rb') as f:
            while True:
                try:
                    tran = dill.load(f)
                    s, a, r, n_s, g, n_g = self.fake_action(
                        tran, mean_tts, ampli, tt_query_dict)
                    self._memor.append([s, a, r, n_s, g, n_g])
                except EOFError:
                    break

        print(f'having totally {len(self._memor)} transitions')

    def fake_a_fn(self, tt, mean_tt, ampli):
        a = max(0, tt-mean_tt)
        a = ampli * a
        a = min(a, self._max_hold)
        return a / self._max_hold

    def construct_new_graph(self, ep, graph, mean_tts, tt_query_dict, ampli):
        '''
            Construct new graph with fake action for DDPG_EG_OFF of DO_NOTHING
        '''
        new_graph = []
        for node in graph:
            real_tt = tt_query_dict[(ep, node.bus_id, node.stop_id, node.time)]
            faked_a = self.fake_a_fn(real_tt, mean_tts[node.stop_id], ampli)
            node = self.__node_feat(
                node.up_or_down, node.augme_info, node.state, faked_a)
            new_graph.append(node)
        return new_graph

    def fake_action(self, tran, mean_tts, ampli, tt_query_dict):
        ep, bus_id, stop_id, ct, s, a, r, n_s, g, n_g = tran
        assert a == 0, 'a should be 0 collected by do-nothing mode'

        real_tt = tt_query_dict[(ep, bus_id, stop_id, ct)]
        faked_a = self.fake_a_fn(real_tt, mean_tts[stop_id], ampli)
        # faked_a = ampli * faked_a
        r = r - self._w * faked_a
        g = self.construct_new_graph(
            ep, g, mean_tts, tt_query_dict, ampli)
        n_g = self.construct_new_graph(
            ep, n_g, mean_tts, tt_query_dict, ampli)
        return s, faked_a, r, n_s, g, n_g

    def __str__(self) -> str:
        return 'DDPG_EG_OFF_DO_NOTHING'

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
