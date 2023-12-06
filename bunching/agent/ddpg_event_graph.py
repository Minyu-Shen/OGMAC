import numpy as np
import torch
import wandb
from torch_geometric.data import Data, Batch
from copy import copy, deepcopy

from .event import Event_Handler
# from .agent import Agent
from .graph_agent import Graph_Agent
from .event_critic_net import Event_Critic_Net
from .net import Actor_Net, Critic_Net


class DDPG_Event_Graph(Graph_Agent):
    def __init__(self, config, agent_config, is_eval) -> None:
        super(DDPG_Event_Graph, self).__init__(config, agent_config, is_eval)
        self.__init_noise_level = agent_config['init_noise_level']
        self.__decay_rate = agent_config['decay_rate']
        self.__gat_state_size = 2
        self.__gat_hidde_size = agent_config['gat_hidde_size']
        self.__layer_init_type = agent_config['layer_init_type']

        if self._is_embed_discr_state:
            embed_discr_state_info = (self._is_embed_discr_state,
                                      self._config.stop_num)
        else:
            embed_discr_state_info = (self._is_embed_discr_state, None)

        self.__event_handl = Event_Handler(config, is_rewar_globa=self._is_rewar_globa, is_rewar_track_globa=True,
                                           is_state_globa=self._is_state_globa, max_hold=self._max_hold,
                                           w=self._w, is_off_policy=True, is_graph=True)
        self.__actor_net = Actor_Net(
            self._state_size, self._hidde_size, embed_discr_state_info, init_type=self.__layer_init_type)

        if not self._is_eval:
            # a list of episode num that need to save model
            self.__check_poins = [1, 100, 150, 200]
            # self.__check_poins = [None]

            self.__ego_criti_net = Critic_Net(
                self._state_size, self._hidde_size, embed_discr_state_info, init_type=self.__layer_init_type)

            self.__event_criti_net = Event_Critic_Net(
                self.__gat_state_size+2, self.__gat_hidde_size, init_type=self.__layer_init_type)

            # target networks
            self.__targe_actor_net = deepcopy(self.__actor_net)
            self.__targe_ego_criti_net = deepcopy(self.__ego_criti_net)
            self.__targe_event_criti_net = deepcopy(self.__event_criti_net)

            # Freeze target networks with respect to optimizers (only update via polyak averaging)
            for param in self.__targe_actor_net.parameters():
                param.requires_grad = False
            for param in self.__targe_ego_criti_net.parameters():
                param.requires_grad = False
            for param in self.__targe_event_criti_net.parameters():
                param.requires_grad = False

            self.__actor_optim = torch.optim.Adam(
                self.__actor_net.parameters(), lr=self._lr)
            self.__ego_critic_optim = torch.optim.Adam(
                self.__ego_criti_net.parameters(), lr=self._lr)
            self.__event_critic_optim = torch.optim.Adam(
                self.__event_criti_net.parameters(), lr=self._lr)

            # maintain update counting
            self.__add_event_count = 0
            # update every self.__update_cycle counts
            self.__updat_cycle = 1
            # maintain noise maker
            self.__noise_level = self.__init_noise_level
        else:
            model = self.load_model(agent_config)
            self.__actor_net.load_state_dict(model)

    def __str__(self) -> str:
        return 'DDPG_EG_ON'

    def reset(self, episode, is_record_wandb=False, is_record_transition=False):
        # headw_varia = super().reset(episode, is_record_wandb)

        if self._is_eval:
            G = self.__event_handl.get_actual_return(self._w, self._gamma)
            headw_varia = super().reset(episode, is_record_wandb, G=G)
        else:
            headw_varia = super().reset(episode, is_record_wandb)

        self.__event_handl.clear_events()
        if not self._is_eval:
            self.__noise_level = self.__decay_rate ** episode * self.__init_noise_level
            if (episode+1) in self.__check_poins:
                self.save_model(self.__actor_net.state_dict(),
                                self._agent_config)

        return headw_varia

    def cal_hold_time(self, snapshot):
        state = copy(snapshot.local_state)
        state.append(snapshot.curr_stop_id)
        actio = self.infer(state)

        # actio = self.infer(snapshot.local_state)

        hold_time = actio.item() * self._max_hold
        # record departure event and log reward
        track_equal_rewar, track_inten_rewar, track_rewar = self.__event_handl.add_event(
            snapshot, actio, hold_time, self._w)
        self.track(track_rewar, track_equal_rewar,
                   track_inten_rewar, hold_time)

        if not self._is_eval:
            self.__add_event_count += 1
            # if accumulated event num is enough, push it to buffer
            if self.__event_handl.get_trans_num_by_bus() > self._batch_size:
                self.__event_handl.push_transition_graph_to_buffer()
            self.learn()
        return hold_time

    def infer(self, state):
        with torch.no_grad():
            a = self.__actor_net(state)
            if not self._is_eval:
                noise = np.random.normal(0, self.__noise_level)
                a = (a + noise).clip(0, 1)
        return a

    def learn(self):
        if self.__add_event_count % self.__updat_cycle != 0 or self.__event_handl.get_buffer_size() < self._batch_size:
            return

        stats, actis, rewas, next_stats, graps, next_graps = self.__event_handl.sample_transition_graph(
            self._batch_size)
        batch_up_data, batch_down_data = self.construct_graph(graps)
        bated_up_data = Batch.from_data_list(batch_up_data)
        bated_down_data = Batch.from_data_list(batch_down_data)

        next_batch_up_data, next_batch_down_data = self.construct_graph(
            next_graps)
        next_bated_up_data = Batch.from_data_list(next_batch_up_data)
        next_bated_down_data = Batch.from_data_list(next_batch_down_data)

        s = torch.tensor(
            stats, dtype=torch.float32).reshape(-1, self._state_size)
        # LongTensor for idx selection
        a = torch.tensor(actis, dtype=torch.float32)
        r = torch.tensor(rewas, dtype=torch.float32)
        n_s = torch.tensor(
            next_stats, dtype=torch.float32).reshape(-1, self._state_size)

        # update critic network
        self.__ego_critic_optim.zero_grad()
        self.__event_critic_optim.zero_grad()

        # current ego estimate
        s_a = torch.concat((s, a.unsqueeze(dim=1)), dim=1)
        for param in self.__ego_criti_net.parameters():
            param.requires_grad = True

        for param in self.__event_criti_net.parameters():
            param.requires_grad = True

        ego_Q = self.__ego_criti_net(s_a)
        # current event estimate
        event_Q = self.__event_criti_net(bated_up_data, bated_down_data)
        # current total estimate
        Q = ego_Q + event_Q

        # Bellman backup for Q function
        targe_imagi_a = self.__targe_actor_net(n_s)  # (batch_size, 1)
        s_targe_imagi_a = torch.concat((n_s, targe_imagi_a), dim=1)
        with torch.no_grad():
            # ego q target
            ego_q_polic_targe = self.__targe_ego_criti_net(s_targe_imagi_a)
            # event q target
            event_q_polic_targe = self.__targe_event_criti_net(
                next_bated_up_data, next_bated_down_data)
            total_targe = ego_q_polic_targe + event_q_polic_targe
            # r is (batch_size, ), need to align with output from NN
            back_up = r.unsqueeze(1) + self._gamma * total_targe
        # MSE loss against Bellman backup
        # Unfreeze Q-network so as to optimize it
        td = Q - back_up
        criti_loss = (td**2).mean()
        # update critic parameters
        criti_loss.backward()
        self.__ego_critic_optim.step()
        self.__event_critic_optim.step()

        # update actor network
        self.__actor_optim.zero_grad()
        imagi_a = self.__actor_net(s)
        s_imagi_a = torch.concat((s, imagi_a), dim=1)
        # Freeze Q-network to save computational efforts
        for param in self.__ego_criti_net.parameters():
            param.requires_grad = False
        for param in self.__event_criti_net.parameters():
            param.requires_grad = False

        Q = self.__ego_criti_net(s_imagi_a)
        actor_loss = -Q.mean()
        actor_loss.backward()
        self.__actor_optim.step()

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.__actor_net.parameters(), self.__targe_actor_net.parameters()):
                p_targ.data.mul_(self._polya)
                p_targ.data.add_((1 - self._polya) * p.data)
            for p, p_targ in zip(self.__ego_criti_net.parameters(), self.__targe_ego_criti_net.parameters()):
                p_targ.data.mul_(self._polya)
                p_targ.data.add_((1 - self._polya) * p.data)
            for p, p_targ in zip(self.__event_criti_net.parameters(), self.__targe_event_criti_net.parameters()):
                p_targ.data.mul_(self._polya)
                p_targ.data.add_((1 - self._polya) * p.data)
