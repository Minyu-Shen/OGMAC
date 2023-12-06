from math import ceil
from simulation.bus import Bus


class Generator:
    def __init__(self, ln_info) -> None:
        self.__ln_info = ln_info

    def generate_buses(self):
        bus_init_link_dict = self.__ln_info['bus_init_link']
        init_link_bus = {}
        for bus_id, init_link in bus_init_link_dict.items():
            init_link_bus[init_link] = Bus(bus_id, self.__ln_info)
        # for bus_id in range(self.__ln_info['bus_num']):
        #     s = init_links[bus_id]
        #     init_link_bus[s] = Bus(bus_id, self.__ln_info)

        return init_link_bus
