# OGMAC

This repository hosts the simulation model and the Offline-GMAC (Offline Graph-based Multi-agent Actor-Critic) implementation featured in the paper titled "Learning to Hold Buses From Historical Uncontrolled Trajectories: An Offline Reinforcement Learning Approach", currently under review.

# Installation

The required packages are listed in `requirements.txt`. Please use the package manager pip to install:

``
pip install -r requirements.txt
``

Additionally, this repository relies on wandb for storing and visualizing training data. Follow the setup guide at [wandb documentation](https://docs.wandb.ai/guides/hosting/how-to-guides/basic-setup) for installation.

# Usage

Please run the program from `main.py` in the bunching folder. Detailed key settings are thoroughly commented.


## Simulator configuration
The configuration regarding the simulator can be set in the `Config` class, which can be instantiated and then passed to the simulator. Note that we only provide the dataset of Line No. 34 of Chengdu (even though a parameter of `34` is passed into the init method); the link and stop information can be found in `34_link.csv` and `34_stop.csv`, respectively.

 The configuration about the control agents can be set in `Agent_Config` and we provide a `sweep` method for grid search over different configuration sets. For the meanings of all the hyperparameters in `Agent_Config`, please refer to our paper or directly contact us via <shenminyu@swufe.edu.cn> if you have any questions.


## Offline-GMAC trained by uncontrolled data
To run the Offline-GMAC trained by uncontrolled data (which is the major contribution of the paper), please first run the simulator for `N` episodes using the `DO_NOTHING` agent to collect the uncontrolled data. The collected data will be stored in the `data` folder. The transition tuples are stored in a file named like `DO_NOTHING_sg_False_rg_True_eg_trans.pickle` where detailed parameters can be found in `Agent_Config` class. You should also manually revise the file name by adding the episode number `N` into it, making it like `DO_NOTHING_day_50_sg_False_rg_True_eg_trans` where `N=50`. The link travel times generated in the simulator (by sampling from a distribution) are stored in a file named `DO_NOTHING_travel_time_eg`. You should also manually revise the file name by adding the `day_N` into the file name.

After collecting the data, you can train the `DDPG_EG_OFF_DO_NOTHING` agent by setting `is_eval` to `False`. After training, the model is automatically saved in the `model` folder and all the hyperparameter settings will also be stored in `config_model_map.json` for further retrieving.

## Forward headway-based (FHC) control method
To run the `FHC` strategy proposed in Daganzo (2009), please note that we should first determine the equilibrium headway for the loop line (i.e., line 34 in this repository); see the Appendix of our paper for more details. In the simulation, it is required that you should set `behav_polic` to `NONLINEAR_UPDATE` in the `Agent_Config` class and run the simulation for multiple episodes and the equilibrium headway from each episode will be stored in the `wandb` database. Then you can set `behav_polic` to `NONLINEAR_FIX_UPDATE` to finally run the FHC strategy using the converged equilibrium headway

## Offline-GMAC trained by FHC-collected data
The process is similar to the one in "Offline-GMAC trained by uncontrolled data" section, except that now you need to run the simulator with the FHC agent to collect the training data. Also, note that travel times are not needed in this case, as we don't need to synthesize the data as in the uncontrolled case.

# Feel free to ask me if you have any questions! :blush:

My email address is: <shenminyu@swufe.edu.cn>
