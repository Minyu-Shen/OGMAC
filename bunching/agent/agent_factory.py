from .nonlinear import Nonlinear
from .do_nothing import Do_Nothing
from .ddpg_event_graph import DDPG_Event_Graph
from .ddpg_event_graph_offline_do_nothing import DDPG_Event_Graph_Off_DN
from .ddpg_event_graph_offline_nonlinear import DDPG_Event_Graph_Off_NL

from .imagination import Imagination
from .td3 import TD3_On
from .td3_offline import TD3_Off
import itertools


class Agent_Factory(object):
    @staticmethod
    def produce_agent(config, agent_config, is_eval):
        if agent_config['agent_name'] == 'DO_NOTHING':
            return Do_Nothing(config, agent_config, is_eval)
        elif agent_config['agent_name'] == 'NONLINEAR':
            return Nonlinear(config, agent_config, is_eval)
        elif agent_config['agent_name'] == 'DDPG_EG_ON':
            return DDPG_Event_Graph(config, agent_config, is_eval)
        elif agent_config['agent_name'] == 'DDPG_EG_OFF_DO_NOTHING':
            return DDPG_Event_Graph_Off_DN(config, agent_config, is_eval)
        elif agent_config['agent_name'] == 'DDPG_EG_OFF_NONLINEAR':
            return DDPG_Event_Graph_Off_NL(config, agent_config, is_eval)
        elif agent_config['agent_name'] == 'IMAGINATION':
            return Imagination(config, agent_config, is_eval)
