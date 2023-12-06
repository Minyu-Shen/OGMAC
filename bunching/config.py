from itertools import accumulate
from collections import defaultdict
from copy import deepcopy as copy
import pandas as pd


class Config:
    def __init__(self, set) -> None:
        self.board_rate = 0.5
        self.aligh_rate = 1.0

        # simulation duration, in seconds
        self.sim_durat = 3600 * 2.0
        # the maximum number of trips for each bus
        # approximately set by rough observations, not accurate
        # used for generate schedules only
        self.max_trip_num = 4
        # simulation time step, in seconds
        self.dt = 1.0
        stop_file_name = 'data/' + str(set) + '_stop.csv'
        link_file_name = 'data/' + str(set) + '_link.csv'
        self.stop_df = pd.read_csv(stop_file_name)
        self.link_df = pd.read_csv(link_file_name)
        bus_init_links = [0, 3, 7, 10, 14, 17, 20, 24, 27]

        # stop number in csv starts from 1
        self.stop_df['stop'] = self.stop_df['stop'] - 1
        self.link_df['link'] = self.link_df['link'] - 1
        self.stop_num = self.stop_df['stop'].max()+1

        bus_init_links = [0, 3, 6, 9, 11, 14,
                          17, 20, 22, 25, 28, 31, 34, 36, 39]
        self.bus_num = len(bus_init_links)

        link_lengs = self.link_df['length'].tolist()
        self.link_lengs = copy(link_lengs[0:self.stop_num])

        self.bus_init_links = copy(bus_init_links[0:self.bus_num])

        self.bus_init_link_dict = {
            bus_id: link_id for bus_id, link_id in enumerate(self.bus_init_links)}

        self.link_num = len(self.link_lengs)

        self.__create_stop_info(set)
        self.__create_link_info(set)
        self.__create_line_info(set)

    def __create_line_info(self, set):
        # stop-link line structure
        stop_next_link = {s: s for s in range(self.stop_num)}
        link_next_stop = {l: l+1 for l in range(self.link_num-1)}
        link_next_stop[self.link_num-1] = 0

        self.ln_info = {'bus_num': self.bus_num, 'stop_num': self.stop_num, 'link_next_stop': link_next_stop,
                        'stop_next_link': stop_next_link, 'board_rate': self.board_rate, 'aligh_rate': self.aligh_rate,
                        'bus_init_link': self.bus_init_link_dict, 'capac': 120}

    def __create_stop_info(self, set):
        # alighting probs for the first 12 stops
        aligh_probs = [0.0135, 0.027, 0.0541, 0.0811, 0.1081, 0.1351,
                       0.1351, 0.1216, 0.1216, 0.0811, 0.0541, 0.0405]
        # alighting probs for the 12-th stop
        aligh_probs.append(1 - sum(aligh_probs))
        self.aligh_probs = tuple(aligh_probs)
        assert sum(aligh_probs) == 1.0, 'Error'

        self.stop_pax_arriv_rate = self.stop_df.set_index('stop')[
            'lambda'].to_dict()

        self.stop_info = defaultdict(dict)
        for s in range(self.stop_num):
            self.stop_info[s]['pax_arriv_rate'] = self.stop_pax_arriv_rate[s]

        xs = list(accumulate(self.link_lengs))
        xs.insert(0, 0)
        xs.pop()
        for s in range(self.stop_num):
            self.stop_info[s]['x'] = xs[s]
            self.stop_info[s]['queue_rule'] = 'unlimited'
            self.stop_info[s]['num_berth'] = 10
            self.stop_info[s]['aligh_probs'] = aligh_probs
            self.stop_info[s]['total_stop_num'] = self.stop_num

    def __create_link_info(self, set):
        end_xs = list(accumulate(self.link_lengs))
        start_xs = [x - y for x, y in zip(end_xs, self.link_lengs)]
        self.link_info = {}

        # convert DataFrame to list of dictionaries
        data = self.link_df.to_dict(orient='records')
        # create dictionary with col1 values as keys and the other three columns as values
        link_info_dict = {d['link']: {'length': d['length'],
                                      'mean_tt': d['mean_tt'], 'cv_tt': d['cv_tt']} for d in data}

        for l, info in link_info_dict.items():
            lengt = info['length']
            mean_speed = lengt / info['mean_tt']
            cv = info['cv_tt']
            self.link_info[l] = {'start_x': start_xs[l], 'end_x': end_xs[l],
                                 'lengt': lengt, 'mean_speed': mean_speed, 'cv': cv}


if __name__ == "__main__":
    config = Config(34)
