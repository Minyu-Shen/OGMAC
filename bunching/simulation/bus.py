from collections import deque, namedtuple
import csv


class Bus:
    '''
    A class used to represent a bus

    Attributes
    ----------
    bus_id : int
        bus id used for identifying bus
    hold_time : float / None
        holding time after finishing service
    traje : dict, time->location information dict
        store bus running trajectory, for plotting TS diagram purpose
    '''

    def __init__(self, bus_id, ln_info) -> None:
        '''
        Parameters
        ---------- 
        bus_id : int
            bus id
        ln_info : dict
            line information dicttionary
        '''
        self.bus_id = bus_id
        self.hold_time = None
        self.traje = {}

        self.__link_next_stop = ln_info['link_next_stop']
        self.__stop_next_link = ln_info['stop_next_link']
        self.__board_rate = ln_info['board_rate']
        self.__aligh_rate = ln_info['aligh_rate']
        self.__stop_num = ln_info['stop_num']
        self.__capac = ln_info['capac']

        # record bus trajecries when entering a link
        self.__link_traje_recor = []

        # bus offset from stop 0
        self.__relat_x = 0.0
        # speed for traversing the link
        self.__speed = None
        # travel time for traversing the link
        self.__trave_time = None
        # current spot type, 'link', or 'stop'
        self.__spot_type = None
        # current spot id
        self.__spot_id = -1
        # current trip number
        self.__trip_no = 0

        # store onboard pax for each destination
        self.__dest_pax = {dest: 0.0 for dest in range(self.__stop_num)}

        # store cumulative pax boardings
        self.cum_pax_board = 0.0
        # store cumulative pax riding time
        self.cum_pax_ride_time = 0.0

    def dump_link_trajectory(self):
        with open('data/DO_NOTHING_travel_time_eg.csv', 'a', encoding='UTF8', newline='') as f:
            # with open('data/DO_NOTHING_travel_time.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            for row in self.__link_traje_recor:
                writer.writerow(row)

    def record_link_trajectory(self, ep, bus_id, stop_id, ct, sampl_time):
        self.__link_traje_recor.append([ep, bus_id, stop_id, ct, sampl_time])

    def get_next_link(self, stop_id):
        return self.__stop_next_link[stop_id]

    def get_next_stop(self, link_id):
        return self.__link_next_stop[link_id]

    def get_occupancy_rate(self):
        return sum(self.__dest_pax.values()) / self.__capac

    def get_remaining_capacity(self):
        return self.__capac - sum(self.__dest_pax.values())

    def accumulate_pax_ride_time(self, dt):
        self.cum_pax_ride_time += sum(self.__dest_pax.values()) * dt

    def board(self, dest, amoun_board):
        self.__dest_pax[dest] += amoun_board
        self.cum_pax_board += amoun_board

    def get_onboard_amount(self, dest):
        return self.__dest_pax[dest]

    def alight(self, dest, dt):
        self.__dest_pax[dest] -= min(self.__dest_pax[dest],
                                     self.__aligh_rate * dt)

    def update_loc(self, ct, spot_type, spot_id, relat_x):
        self.__spot_type = spot_type
        self.__spot_id = spot_id
        self.__relat_x = relat_x
        occup_rate = self.get_occupancy_rate()
        self.traje[ct] = {'trip_no': self.__trip_no, 'spot_type': spot_type,
                          'spot_id': spot_id, 'relat_x': relat_x, 'occup_rate': occup_rate}

    def get_bus_loc_info(self):
        return {'spot_type': self.__spot_type, 'spot_id': self.__spot_id, 'x': self.__relat_x}

    def get_next_loc_on_link(self, link_end_x, dt):
        x = min(link_end_x, self.__relat_x + self.__speed*dt)
        return x

    def set_speed(self, speed):
        self.__speed = speed

    def reduce_hold_time(self, dt):
        self.hold_time -= dt
        if self.hold_time <= 0.0:
            self.hold_time = None

    def count_trip_no(self):
        self.__trip_no += 1

    def get_trip_no(self):
        return self.__trip_no

    @property
    def board_rate(self):
        return self.__board_rate

    @ property
    def relat_x(self):
        return self.__relat_x

    def set_travel_time(self, trave_time):
        self.__trave_time = trave_time

    def get_travel_time(self):
        return self.__trave_time
