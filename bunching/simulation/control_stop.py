from bunching.simulation.stop import Stop


class Control_Stop(Stop):
    def __init__(self, info) -> None:
        super().__init__(info)

    def check_in(self):
        pass
