import numpy as np
from numpy.random import lognormal
import csv


class Link:
    def __init__(self, link_id, info) -> None:
        self.__link_id = link_id
        self.__lengt = info["lengt"]
        self.__mean_speed = info["mean_speed"]
        self.__cv = info["cv"]
        self.__start_x = info["start_x"]
        self.__end_x = info["end_x"]
        self.reset()

    def get_mean_travel_time(self):
        return self.__lengt / self.__mean_speed

    def enter_bus(self, ct, bus, ep):
        self.__buses.append(bus)
        # sample a lognormal travel time
        mu = self.__lengt / self.__mean_speed
        sigma = self.__cv * mu
        norma_std = np.sqrt(np.log(1 + (sigma / mu) ** 2))
        norma_mean = np.log(mu) - norma_std ** 2 / 2
        sampl_time = lognormal(norma_mean, norma_std)
        sampl_speed = self.__lengt / sampl_time
        bus_id = bus.bus_id
        stop_id = bus.get_bus_loc_info()['spot_id']
        # trip_no = bus.trip_no
        # set bus speed
        bus.set_speed(sampl_speed)
        bus.set_travel_time(sampl_time)
        if stop_id >= 0:
            bus.record_link_trajectory(ep, bus_id, stop_id, ct, sampl_time)

        # record sampled travel time into csv
        # if stop_id >= 0:
        #     with open('historical_no_a_travel_time.csv', 'a', encoding='UTF8', newline='') as f:
        #         writer = csv.writer(f)
        #         writer.writerow([ep, bus_id, stop_id, ct, sampl_time])

        # update bus location information
        bus.update_loc(ct, "link", self.__link_id, self.__start_x)

    def reset(self):
        self.__buses = []

    def forward(self, ct, dt):
        leave_buses = []
        for bus in self.__buses:
            # if not True:
            if bus.hold_time is not None and bus.hold_time > 0:
                # holding operations
                bus.reduce_hold_time(dt)
                bus.update_loc(ct, "hold", self.__link_id, self.__start_x)
            else:
                # traveling operations
                x = bus.get_next_loc_on_link(self.__end_x, dt)
                bus.update_loc(ct, "link", self.__link_id, x)
                if x >= self.__end_x:
                    leave_buses.append(bus)

        self.__buses = [bus for bus in self.__buses if bus not in leave_buses]
        return leave_buses
