import numpy as np
from .agent import Agent
import wandb
from .event import Event_Handler
from sympy import Symbol, nsolve
from .stable_schedules import find_stable_H_and_schedules
from .order_stats import find_order_stat_ratio


class Nonlinear(Agent):
    def __init__(self, config, agent_config, is_eval):
        super(Nonlinear, self).__init__(config, agent_config, is_eval)

        self.config = config
        self.__behav_polic = agent_config['behav_polic']
        self.__is_graph = agent_config['is_graph']
        self.__pertu_range = agent_config['pertu_range']

        self.__type_dict = {'behav_polic': self.__behav_polic, 'is_state_globa': self._is_state_globa,
                            'is_rewar_globa': self._is_rewar_globa, 'is_graph': self.__is_graph,
                            'pertu_range': self.__pertu_range}

        self.__event_handl = Event_Handler(config, is_rewar_globa=self._is_rewar_globa, is_rewar_track_globa=False,
                                           is_state_globa=self._is_state_globa, max_hold=self._max_hold, w=self._w, is_graph=self.__is_graph)

        self.__cal_stop_beta()
        self.__stop_holds = {s: [] for s in range(config.stop_num)}
        self.__alpha = agent_config['alpha']
        self.__slack = agent_config['slack']

        if self.__behav_polic == 'NONLINEAR_FIX_UPDATE':
            # H found by simulation
            self.__H, _ = find_stable_H_and_schedules(
                'NONLINEAR', self.__alpha)

        elif self.__behav_polic == 'NONLINEAR_UPDATE':
            self.__set_equilibrium_H()

        print(self.__alpha, ':',  self.__H)

    def __cal_stop_beta(self):
        # create beta for each stop
        self.__stop_beta = {}
        # print('aligh:', config.aligh_probs)
        # print('arriv rate:', config.stop_pax_arriv_rate)
        for stop_id, info in self.config.stop_info.items():
            arriv_rate_in_sec = (info['pax_arriv_rate'] / 60.0)
            prev_stop_ids = [(stop_id - i - 1) %
                             self.config.stop_num for i in range(len(self.config.aligh_probs))]
            cum_aligh = 0.0
            for idx, prev_stop_id in enumerate(prev_stop_ids):
                cum_aligh += self.config.stop_pax_arriv_rate[prev_stop_id] / \
                    60.0 * self.config.aligh_probs[idx]
            # board_beta dominates aligh_beta
            board_beta = arriv_rate_in_sec / self.config.board_rate
            aligh_beta = cum_aligh / self.config.aligh_rate
            order_ratio = find_order_stat_ratio(board_beta, aligh_beta)
            self.__stop_beta[stop_id] = order_ratio * board_beta
            # self.__stop_beta[stop_id] = max(board_beta, aligh_beta)

    def __set_equilibrium_H(self):
        H = Symbol('H')
        avg_trip_time = 0

        for _, info in self.config.link_info.items():
            link_time = info['lengt'] / info['mean_speed']
            avg_trip_time += link_time
        for stop, _ in self.config.stop_info.items():
            beta = self.__stop_beta[stop]
            avg_trip_time += beta * H
        for stop in range(self.config.stop_num):
            mean_hold = np.mean(self.__stop_holds[stop]) if len(
                self.__stop_holds[stop]) > 0 else 0
            avg_trip_time += mean_hold

            # print(self.__stop_holds[stop])
            # print(mean_hold)

        sched_headw = nsolve(avg_trip_time/self.config.bus_num - H, H, 1)
        sched_headw = float(sched_headw)
        self.__H = sched_headw
        # avg_trip_time = avg_trip_time.subs(H, config.sched_headw)
        # avg_trip_time = float(avg_trip_time)

    def __str__(self) -> str:
        return 'NONLINEAR'

    def sample_H(self):
        return np.random.uniform(0.01, 1.01), self.__H

    def reset(self, episode, is_record_wandb=False, is_record_transition=False):
        G = self.__event_handl.get_actual_return(self._w, self._gamma)
        super().reset(episode, is_record_wandb, G=G)
        # wandb.log({'G': G})
        # at the end of each episode, write it to csv
        if is_record_transition:
            self.__event_handl.write_transition_to_file(self.__type_dict)

        self.__event_handl.clear_events()

        # update interacting policy parameter
        if self.__behav_polic == 'NONLINEAR_FIX':
            pass
        elif self.__behav_polic == 'NONLINEAR_UPDATE':
            self.__set_equilibrium_H()
            self.__stop_holds = {s: [] for s in range(self.config.stop_num)}
            wandb.log({'H': self.__H, 'G': G})

    def cal_hold_time(self, snapshot):
        ct = snapshot.ct
        assert ct == snapshot.last_depar_times[-1]
        last_depar_time = snapshot.last_depar_times[-2]
        if last_depar_time == 0:
            hold_time = 0
        else:
            devia = self.__H - (ct - last_depar_time)
            pertu = np.random.uniform(-self.__pertu_range,
                                      self.__pertu_range) if self.__pertu_range > 0 else 0
            devia += pertu
            devia += self.__slack
            hold_time = max(0, self.__alpha*devia)
            hold_time = min(self._max_hold, hold_time)

        self.__stop_holds[snapshot.curr_stop_id].append(hold_time)

        # record departure event and log reward
        actio = hold_time / self._max_hold
        track_equal_rewar, track_inten_rewar, track_rewar = self.__event_handl.add_event(
            snapshot, actio, hold_time, self._w)
        self.track(track_rewar, track_equal_rewar,
                   track_inten_rewar, hold_time)

        return hold_time
