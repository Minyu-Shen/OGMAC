from .stop import Stop
from .link import Link
from .generator import Generator
from .snapshot import Snapshot


class Simulator:
    def __init__(self, config, agent=None, is_recor_link_traje=False) -> None:
        # indicate if the link trajectory of each bus should be recorded to csv
        self.__is_recor_link_traje = is_recor_link_traje
        # RL control agent
        self.__agent = agent
        self.__config = config

        # simulation
        self.__dt = config.dt
        self.__sim_durat = config.sim_durat
        self.__gener = Generator(config.ln_info)

        # init bus stops
        self.__stops = {}
        for stop_id, info in config.stop_info.items():
            stop = Stop(stop_id, info)
            self.__stops[stop_id] = stop

        # init links
        self.__links = {}
        for link_id, info in config.link_info.items():
            link = Link(link_id, info)
            self.__links[link_id] = link

        self.reset(episode=0)

    def get_buses_for_plot(self):
        return self.__buses

    def reset(self, episode):
        self.__episode = episode
        self.__ct = 0.0
        for _, link in self.__links.items():
            link.reset()
        for _, stop in self.__stops.items():
            stop.reset()

        self.__buses = []
        init_link_bus = self.__gener.generate_buses()
        for init_link, bus in init_link_bus.items():
            self.__buses.append(bus)
            self.__links[init_link].enter_bus(self.__ct, bus, self.__episode)

    def simulate(self):
        while True:
            self.step()
            if self.__ct > self.__sim_durat:
                self.record()
                if self.__is_recor_link_traje:
                    self.dump_bus_link_trajectory()
                break

    def record(self):
        # when simulation ends, record the wait time and arrival of each stop
        self.__agent.record_wait_time(self.get_wait_time_each_stop())
        self.__agent.record_arrival(self.get_arrival_each_stop())
        self.__agent.record_ride_time(self.get_ride_time_each_bus())
        self.__agent.record_boarding(self.get_boarding_each_bus())

    def get_boarding_each_bus(self):
        boarding_dict = {}
        for bus in self.__buses:
            boarding_dict[bus.bus_id] = bus.cum_pax_board
        return boarding_dict

    def get_ride_time_each_bus(self):
        ride_time_dict = {}
        for bus in self.__buses:
            ride_time_dict[bus.bus_id] = bus.cum_pax_ride_time
        return ride_time_dict

    def get_arrival_each_stop(self):
        stop_arrival_dict = {}
        for stop_id, stop in self.__stops.items():
            stop_arrival_dict[stop_id] = stop.pax_total_arriv
        return stop_arrival_dict

    def get_wait_time_each_stop(self):
        stop_wait_time_dict = {}
        for stop_id, stop in self.__stops.items():
            stop_wait_time_dict[stop_id] = stop.pax_total_wait_time
        return stop_wait_time_dict

    def dump_bus_link_trajectory(self):
        for bus in self.__buses:
            bus.dump_link_trajectory()

    def step(self):
        # link operations
        for link_id, link in self.__links.items():
            leave_buses = link.forward(self.__ct, self.__dt)
            self.__transfer(leave_buses, link_id=link_id, stop_id=None)

        # stop operations
        for stop_id, stop in self.__stops.items():
            leave_buses = stop.operation(self.__ct, self.__dt)
            self.__transfer(leave_buses, link_id=None, stop_id=stop_id)

        # accumulate pax riding time for each time step
        for bus in self.__buses:
            bus.accumulate_pax_ride_time(self.__dt)

        self.__ct += self.__dt

    def __transfer(self, leave_buses, link_id=None, stop_id=None):
        for bus in leave_buses:
            if link_id is not None:
                next_stop_id = bus.get_next_stop(link_id)
                self.__stops[next_stop_id].enter_bus(self.__ct, bus)
                # if one trip (loop) is finished, count trip_no += 1
                if next_stop_id == 0:
                    bus.count_trip_no()
            # do action when entering the next link
            if stop_id is not None:
                next_link_id = bus.get_next_link(stop_id)
                self.__links[next_link_id].enter_bus(
                    self.__ct, bus, self.__episode)

                # if the agent is IMAGINATION, then the hold time is the deviation from the mean travel time
                if str(self.__agent) == 'IMAGINATION':
                    mean_tt = self.__links[next_link_id].get_mean_travel_time()
                    tt_in_advan = bus.get_travel_time()
                    # delta = bus.get_travel_time() - mean_tt
                    snapshot = self.__take_snapshot(
                        bus.bus_id, stop_id, bus.get_trip_no())
                    hold_time = self.__agent.get_hold_time(
                        mean_tt, tt_in_advan, snapshot)
                else:
                    # take a snapshot and determine the holding time
                    occup_rate = bus.get_occupancy_rate()
                    snapshot = self.__take_snapshot(
                        bus.bus_id, stop_id, bus.get_trip_no())
                    hold_time = self.__agent.cal_hold_time(snapshot)
                bus.hold_time = hold_time
                self.__agent.track_headway_deviations(snapshot)

    def __take_snapshot(self, curr_bus_id, curr_stop_id, curr_trip_no):
        '''Take a snapshot of the whole system when a bus departs from the stop'''
        snapshot = Snapshot(self.__episode, self.__ct, curr_bus_id, curr_stop_id,
                            curr_trip_no, self.__buses, self.__stops, self.__config)
        return snapshot
