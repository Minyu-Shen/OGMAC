import numpy as np
import torch
import wandb
from torch_geometric.data import Data, Batch
import random

from .event import Event_Handler
# from .agent import Agent
from .graph_agent import Graph_Agent
from .event_critic_net import Event_Critic_Net
from .net import Actor_Net, Critic_Net
from copy import copy, deepcopy


class DDPG_Event_Graph_Off(Graph_Agent):
    def __init__(self, config, agent_config, is_eval) -> None:
        super(DDPG_Event_Graph_Off, self).__init__(
            config, agent_config, is_eval)
        self.__gat_state_size = 2
        self.__gat_hidde_size = agent_config['gat_hidde_size']
        self.__layer_init_type = agent_config['layer_init_type']

        self.__event_handl = Event_Handler(config, is_rewar_globa=self._is_rewar_globa, is_rewar_track_globa=True,
                                           is_state_globa=self._is_state_globa, max_hold=self._max_hold,
                                           w=self._w, is_off_policy=True, is_graph=True)
        if self._is_embed_discr_state:
            embed_discr_state_info = (self._is_embed_discr_state,
                                      self._config.stop_num)
        else:
            embed_discr_state_info = (self._is_embed_discr_state, None)

        self._actor_net = Actor_Net(
            self._state_size, self._hidde_size, embed_discr_state_info, init_type=self.__layer_init_type)

        if not self._is_eval:
            self._memor = []
            # self.__max_off_iter_step = 40000

            self.__ego_criti_net = Critic_Net(
                self._state_size, self._hidde_size, embed_discr_state_info, init_type=self.__layer_init_type)

            self.__event_criti_net = Event_Critic_Net(
                self.__gat_state_size+2, self.__gat_hidde_size, init_type=self.__layer_init_type)

            # target networks
            self.__targe_actor_net = deepcopy(self._actor_net)
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
                self._actor_net.parameters(), lr=self._lr)
            self.__ego_critic_optim = torch.optim.Adam(
                self.__ego_criti_net.parameters(), lr=self._lr)
            self.__event_critic_optim = torch.optim.Adam(
                self.__event_criti_net.parameters(), lr=self._lr)

    def form_memory(self):
        raise NotImplementedError

    # def __str__(self) -> str:
    #     return 'DDPG_EG_OFF'

    def reset(self, episode, is_record_wandb=False, is_record_transition=False):
        # write to wandb
        G = self.__event_handl.get_actual_return(self._w, self._gamma)
        super().reset(episode, is_record_wandb, G=G)

        # if self._is_eval and is_record_wandb:
        #     wandb.log({'mc_return': G})

        self.__event_handl.clear_events()

    def cal_hold_time(self, snapshot):
        if self._is_state_globa:
            actio = self.infer(snapshot.globa_relat_state)
        else:
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
        return hold_time

    def infer(self, state):
        with torch.no_grad():
            a = self._actor_net(state)
        return a

    def offline_learn(self):
        if not self._is_eval:
            # wandb.init(project='offline_train', config=self._agent_config)
            wandb.init(project='offline_day', config=self._agent_config)

        one_epoch_batch_num = len(self._memor) // self._batch_size
        total_iter_step = one_epoch_batch_num * 50

        epoch_actor_loses = []
        epoch_criti_loses = []
        epoch_ego_Qs = []
        epoch_event_Qs = []
        epoch_total_Qs = []

        for i in range(total_iter_step):
            # for i in range(self.__max_off_iter_step):
            if (i+1) % 1000 == 0:
                print('------ offline training iteration', (i+1), '----------')

            stats = []
            actis = []
            rewas = []
            next_stats = []
            graps = []
            next_graps = []
            samps = random.sample(self._memor, self._batch_size)
            for sampl in samps:
                state, actio, rewar, next_state, graph, next_graph = sampl
                stats.append(state)
                actis.append(actio)
                rewas.append(rewar)
                next_stats.append(next_state)
                graps.append(graph)
                next_graps.append(next_graph)

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

            # 1. update critic network
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
            total_Q = ego_Q + event_Q
            # total_Q = ego_Q

            # Bellman backup for Q function
            targe_imagi_a = self.__targe_actor_net(n_s)  # (batch_size, 1)
            s_targe_imagi_a = torch.concat((n_s, targe_imagi_a), dim=1)
            with torch.no_grad():
                # ego q target
                ego_q_polic_targe = self.__targe_ego_criti_net(s_targe_imagi_a)
                # event q target
                event_q_polic_targe = self.__targe_event_criti_net(
                    next_bated_up_data, next_bated_down_data)

                # combined target
                total_targe = ego_q_polic_targe + event_q_polic_targe
                # total_targe = ego_q_polic_targe

                # r is (batch_size, ), need to align with output from NN
                back_up = r.unsqueeze(1) + self._gamma * total_targe
            # MSE loss against Bellman backup
            # Unfreeze Q-network so as to optimize it
            td = total_Q - back_up
            # l1_loss = torch.nn.SmoothL1Loss(reduction='mean')
            # criti_loss = l1_loss(total_Q, back_up)
            criti_loss = (td**2).mean()

            # update critic parameters
            criti_loss.backward()
            self.__ego_critic_optim.step()
            self.__event_critic_optim.step()

            # 2. update actor network
            self.__actor_optim.zero_grad()
            imagi_a = self._actor_net(s)
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

            epoch_actor_loses.append(actor_loss.item())
            epoch_criti_loses.append(criti_loss.item())
            epoch_ego_Qs.append(ego_Q.mean().item())
            epoch_event_Qs.append(event_Q.mean().item())
            epoch_total_Qs.append(total_Q.mean().item())

            if not self._is_eval and (i) % one_epoch_batch_num == 0:
                wandb.log({'criti_loss': np.mean(epoch_criti_loses),
                           'actor_loss': np.mean(epoch_actor_loses),
                           'total_Q': np.mean(epoch_total_Qs),
                           'ego_Q': np.mean(epoch_ego_Qs),
                           'event_Q': np.mean(epoch_event_Qs)})
                epoch_actor_loses = []
                epoch_criti_loses = []
                epoch_ego_Qs = []
                epoch_event_Qs = []
                epoch_total_Qs = []

            # if not self._is_eval and (i+1) % 1 == 0:
            #     wandb.log({'criti_loss': criti_loss.item(),
            #                'actor_loss': actor_loss.item(),
            #                'total_Q': total_Q.mean().item(),
            #                'ego_Q': ego_Q.mean().item(),
            #                'event_Q': event_Q.mean().item()})

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self._actor_net.parameters(), self.__targe_actor_net.parameters()):
                    p_targ.data.mul_(self._polya)
                    p_targ.data.add_((1 - self._polya) * p.data)
                for p, p_targ in zip(self.__ego_criti_net.parameters(), self.__targe_ego_criti_net.parameters()):
                    p_targ.data.mul_(self._polya)
                    p_targ.data.add_((1 - self._polya) * p.data)
                for p, p_targ in zip(self.__event_criti_net.parameters(), self.__targe_event_criti_net.parameters()):
                    p_targ.data.mul_(self._polya)
                    p_targ.data.add_((1 - self._polya) * p.data)

# # epochs
#         for i in range(10):
#             random.shuffle(self._memor)
#             batch_num = int(len(self._memor) / self._batch_size)
#             for j in range(batch_num):
#                 stats = []
#                 actis = []
#                 rewas = []
#                 next_stats = []
#                 graps = []
#                 next_graps = []
#                 batch = self._memor[j*self._batch_size:(j+1)*self._batch_size]
#                 for sampl in batch:
#                     state, actio, rewar, next_state, graph, next_graph = sampl
#                     stats.append(state)
#                     actis.append(actio)
#                     rewas.append(rewar)
#                     next_stats.append(next_state)
#                     graps.append(graph)
#                     next_graps.append(next_graph)

#                 batch_up_data, batch_down_data = self.construct_graph(graps)
#                 bated_up_data = Batch.from_data_list(batch_up_data)
#                 bated_down_data = Batch.from_data_list(batch_down_data)

#                 next_batch_up_data, next_batch_down_data = self.construct_graph(
#                     next_graps)
#                 next_bated_up_data = Batch.from_data_list(next_batch_up_data)
#                 next_bated_down_data = Batch.from_data_list(
#                     next_batch_down_data)

#                 s = torch.tensor(
#                     stats, dtype=torch.float32).reshape(-1, self._state_size)
#                 # LongTensor for idx selection
#                 a = torch.tensor(actis, dtype=torch.float32)
#                 r = torch.tensor(rewas, dtype=torch.float32)
#                 n_s = torch.tensor(
#                     next_stats, dtype=torch.float32).reshape(-1, self._state_size)

#                 # update critic network
#                 self.__ego_critic_optim.zero_grad()
#                 self.__event_critic_optim.zero_grad()

#                 # current ego estimate
#                 s_a = torch.concat((s, a.unsqueeze(dim=1)), dim=1)
#                 for param in self.__ego_criti_net.parameters():
#                     param.requires_grad = True

#                 for param in self.__event_criti_net.parameters():
#                     param.requires_grad = True

#                 ego_Q = self.__ego_criti_net(s_a)
#                 # current event estimate
#                 event_Q = self.__event_criti_net(
#                     bated_up_data, bated_down_data)
#                 # current total estimate
#                 total_Q = ego_Q + event_Q

#                 # Bellman backup for Q function
#                 targe_imagi_a = self.__targe_actor_net(n_s)  # (batch_size, 1)
#                 s_targe_imagi_a = torch.concat((n_s, targe_imagi_a), dim=1)
#                 with torch.no_grad():
#                     # ego q target
#                     ego_q_polic_targe = self.__targe_ego_criti_net(
#                         s_targe_imagi_a)
#                     # event q target
#                     event_q_polic_targe = self.__targe_event_criti_net(
#                         next_bated_up_data, next_bated_down_data)
#                     total_targe = ego_q_polic_targe + event_q_polic_targe
#                     # r is (batch_size, ), need to align with output from NN
#                     back_up = r.unsqueeze(1) + self._gamma * total_targe
#                 # MSE loss against Bellman backup
#                 # Unfreeze Q-network so as to optimize it
#                 td = total_Q - back_up
#                 criti_loss = (td**2).mean()
#                 # update critic parameters
#                 criti_loss.backward()
#                 self.__ego_critic_optim.step()
#                 self.__event_critic_optim.step()

#                 # update actor network
#                 self.__actor_optim.zero_grad()
#                 imagi_a = self._actor_net(s)
#                 s_imagi_a = torch.concat((s, imagi_a), dim=1)
#                 # Freeze Q-network to save computational efforts
#                 for param in self.__ego_criti_net.parameters():
#                     param.requires_grad = False
#                 for param in self.__event_criti_net.parameters():
#                     param.requires_grad = False

#                 Q = self.__ego_criti_net(s_imagi_a)
#                 actor_loss = -Q.mean()
#                 actor_loss.backward()
#                 self.__actor_optim.step()

#                 # if not self._is_eval and (i+1) % 100 == 0:
#                 wandb.log({'criti_loss': criti_loss.item(),
#                            'actor_loss': actor_loss.item(),
#                            'total_Q': total_Q.mean().item(),
#                            'ego_Q': ego_Q.mean().item(),
#                            'event_Q': event_Q.mean().item()})

#                 # Finally, update target networks by polyak averaging.
#                 with torch.no_grad():
#                     for p, p_targ in zip(self._actor_net.parameters(), self.__targe_actor_net.parameters()):
#                         p_targ.data.mul_(self._polya)
#                         p_targ.data.add_((1 - self._polya) * p.data)
#                     for p, p_targ in zip(self.__ego_criti_net.parameters(), self.__targe_ego_criti_net.parameters()):
#                         p_targ.data.mul_(self._polya)
#                         p_targ.data.add_((1 - self._polya) * p.data)
#                     for p, p_targ in zip(self.__event_criti_net.parameters(), self.__targe_event_criti_net.parameters()):
#                         p_targ.data.mul_(self._polya)
#                         p_targ.data.add_((1 - self._polya) * p.data)
