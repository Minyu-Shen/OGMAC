from simulation.simulator import Simulator
from simulation.visualize.visualize import plot_time_space_diagram
from config import Config
from agent.agent_factory import Agent_Factory
from agent.agent_config import Agent_Config
import wandb
from arena import set_seed


set_seed(0)
# line 34 of Chengdu
config = Config(set=34)

# specify if need to record the transition tuple
# used when generating uncontrolled data collected by 'DO_NOTHING' or 'NONLINEAR' agent
is_recor_trans = True

# speficy if need to record the link travel times
# used when synthesizing "controlled" data from uncontrolled data collected by 'DO_NOTHING' agent
# see Algorithm 2 in the paper
is_recor_link_traje = True

# speficy if need to record the data to wandb
is_recor_wandb = False

# speficy if the agent is in evaluation mode
# if False:
#   for offline agents, the agent will be trained offline, the training is done in the "__init__" function of the agent
#   for online agents, the agent will collect data and be trained, make sure ep_num > 0 in this case for online training
# if True:
#   for both offline and online agents, the agent will be evaluated
# for do-nothing agent and the revised version of Daganzo (2009) in "nonlinear.py", it does not matter if is_eval is True or False
is_eval = False

# 1. no-control agent
agent_name = 'DO_NOTHING'

# 2. revised version in Daganzo (2009)
# agent_name = 'NONLINEAR'

# 3. online GMAC
# agent_name = 'DDPG_EG_ON'

# 4. offline GMAC trained by the data collected by 'DO_NOTHING' agent
# agent_name = 'DDPG_EG_OFF_DO_NOTHING'

# 5. offline GMAC trained by the data collected by 'NONLINEAR' agent
# agent_name = 'DDPG_EG_OFF_NONLINEAR'

# the project name for wandb
project_name = 'test'

sweep = Agent_Config(agent_name).get_sweep()
ep_num = 1

for agent_config in sweep:
    headw_varis = []
    print('total sweep:', len(sweep), ', current sweep:',
          sweep.index(agent_config)+1)
    print('agent_name:', agent_config['agent_name'], 'sweeping:', agent_config)
    agent = Agent_Factory.produce_agent(config, agent_config, is_eval)
    if is_recor_wandb:
        wandb.init(project=project_name, config=agent_config)
    simulator = Simulator(config, agent, is_recor_link_traje)
    for ep in range(ep_num):
        simulator.reset(episode=ep)
        simulator.simulate()
        headw_varia = agent.reset(
            episode=ep, is_record_wandb=is_recor_wandb, is_record_transition=is_recor_trans)
        headw_varis.append(headw_varia)

    # plotting the time-space diagram for the last episode
    # plot_time_space_diagram(simulator.get_buses_for_plot(), config)
    wandb.finish()
