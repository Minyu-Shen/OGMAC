import torch
import wandb
import csv
import random
import ast
import pandas as pd
import numpy as np
import os
import dill

from .event import Event_Handler
from .agent import Agent
from .approximator import MLP


class TD3_Off(Agent):
    def __init__(self, config, is_eval) -> None:
        super(TD3_Off, self).__init__(config)
        # indicate whether it is in evaluation mode. if not, it is in training mode
        self._is_eval = is_eval

        self.__behav_policy = 'DO_NOTHING'
        self.__behav_policy = 'NONLINEAR'
        self._is_state_globa = False
        self._is_rewar_globa = False
        self._w = 0.05
        self._gamma = 0.9

        # actor update is delayed for some rounds
        self.__actor_delay_round = 5

        if self._is_state_globa == True:
            self._state_size = 6
        else:
            self._state_size = 2

        file = self.__behav_policy
        file += '_state_globa_' + str(self._is_state_globa)
        file += '_rewar_globa_' + str(self._is_rewar_globa)
        if self.__behav_policy == 'NONLINEAR':
            self.__pertu_range = 0
            file += '_pertu_' + str(self.__pertu_range)
        file += '_transition.csv'
        self.__tran_file = file
        print(self.__tran_file)

        if self.__behav_policy == 'DO_NOTHING':
            self.__tt_file = 'DO_NOTHING_travel_time.csv'

        self._max_hold = 60.0
        self.__actor_net = MLP(self._state_size, 1,
                               hidde_size=(64,), outpu='sigmoid')
        self.__criti_net = MLP(self._state_size+1, 1,
                               hidde_size=(64,), outpu='logits')
        self.__criti_net_dual = MLP(
            self._state_size+1, 1, hidde_size=(64,), outpu='logits')

        self.__event_handl = Event_Handler(config, is_rewar_globa=False, is_rewar_track_globa=True,
                                           is_state_globa=False, max_hold=self._max_hold, w=self._w, is_off_policy=True)

        # return to main function (wandb) for plotting
        self.hyper_paras = {'agent_name': 'TD3_OFF_'+self.__behav_policy, 'w': self._w,
                            'gamma': self._gamma, 'is_eval': self._is_eval}

        if not self._is_eval:
            self.__memor = []
            self._polya = 0.995
            # amplification factor, dynamically determined by sampled travel times
            self.__ampli = None
            self.__max_off_iter_step = 30000
            self.__targe_actor_net = MLP(
                self._state_size, 1, hidde_size=(64,), outpu='sigmoid')
            self.__targe_actor_net.load_state_dict(
                self.__actor_net.state_dict())

            self.__targe_criti_net = MLP(
                self._state_size+1, 1, hidde_size=(64, ), outpu='logits')
            self.__targe_criti_net.load_state_dict(
                self.__criti_net.state_dict())

            self.__targe_criti_net_dual = MLP(
                self._state_size+1, 1, hidde_size=(64, ), outpu='logits')
            self.__targe_criti_net_dual.load_state_dict(
                self.__criti_net_dual.state_dict())

            # Freeze target networks with respect to optimizers (only update via polyak averaging)
            for param in self.__targe_actor_net.parameters():
                param.requires_grad = False
            for param in self.__targe_criti_net.parameters():
                param.requires_grad = False
            for param in self.__targe_criti_net_dual.parameters():
                param.requires_grad = False

            self._batch_size = 64
            # counting for critic update
            self.__criti_updat_count = 0
            self.__actor_optim = torch.optim.Adam(
                self.__actor_net.parameters(), lr=5e-3)
            self.__criti_optim = torch.optim.Adam(
                self.__criti_net.parameters(), lr=5e-3)
            self.__criti_optim_dual = torch.optim.Adam(
                self.__criti_net_dual.parameters(), lr=5e-3)

            self.form_memory()
            self.offline_learn()
        else:
            agent_name = self.__str__() + '_' + self.__behav_policy + '_iter_' + str(20000)
            model = self.check_point_out(agent_name, self._w, None)
            self.__actor_net.load_state_dict(model)

    def form_memory(self):
        if self.__behav_policy == 'DO_NOTHING':
            self.fake_action_transitions_and_add_to_memory(
                self._config.link_lengs, self._config.mean_speed)

        elif self.__behav_policy == 'NONLINEAR':
            with open(self.__tran_file, 'r') as f:
                reade = csv.reader(f, delimiter=',')
                for row in reade:
                    _, _, _, _, s, a, r, n_s = row
                    r = str(float(r) - self._w * float(a))
                    self.__memor.append([s, a, r, n_s])

    def fake_action_transitions_and_add_to_memory(self, link_lengs, mean_speed):
        mean_tts = [x / mean_speed for x in link_lengs]
        tran_header = ['ep', 'bus_id', 'stop_id', 'ct', 's', 'a', 'r', 'n_s']
        trans_df = pd.read_csv(self.__tran_file, names=tran_header)
        tt_header = ['ep', 'bus_id', 'stop_id', 'ct', 'tt']
        tt_df = pd.read_csv(self.__tt_file, names=tt_header)
        df = pd.merge(trans_df, tt_df, on=['ep', 'bus_id', 'stop_id', 'ct'])

        df['dev'] = df.apply(lambda row: (
            row.tt - mean_tts[row.stop_id])**2, axis=1)
        mean_sigma = np.sqrt(df['dev'].mean())
        self.__ampli = self._max_hold / (3*mean_sigma)
        print(self.__ampli)

        def infer_a(tt, mean_tt):
            a = max(0, tt-mean_tt)
            a = self.__ampli * a
            a = min(a, self._max_hold)
            return a / self._max_hold
        # dev = (sampled travel time - mean_tt) / mean_tt
        df['a'] = df.apply(lambda row: infer_a(
            row.tt, mean_tts[row.stop_id]), axis=1)
        df['r'] = df.apply(lambda row: row.r + self._w*(-row.a), axis=1)
        df[['s', 'a', 'r', 'n_s']].to_csv(
            'fake_transition.csv', header=None, index=None, mode='w')

        with open('fake_transition.csv', 'r') as f:
            reade = csv.reader(f, delimiter=',')
            for row in reade:
                self.__memor.append(row)

    def reset(self, episode, is_record_wandb=False, is_record_transition=False):
        super().reset(episode, is_record_wandb)

    def set_wandb_watcher(self):
        wandb.watch(self.__criti_net, log='all', log_freq=10)

    def __str__(self) -> str:
        return 'DDPG_OFF'

    def cal_hold_time(self, snapshot):
        actio = self.infer(snapshot.local_state)
        hold_time = actio.item() * self._max_hold
        # record departure event and log reward
        track_equal_rewar, track_inten_rewar, track_rewar = self.__event_handl.add_event(
            snapshot, actio, hold_time, self._w)
        self.track(track_rewar, track_equal_rewar,
                   track_inten_rewar, hold_time)
        return hold_time

    def infer(self, state):
        with torch.no_grad():
            a = self.__actor_net(state)
        return a

    def offline_learn(self):
        wandb.init(project='offline', config=self.hyper_paras)
        for i in range(self.__max_off_iter_step):
            if (i+1) % 5000 == 0:
                print('------ offline training iteration', i+1, '----------')
                agent_name = self.__str__() + '_' + self.__behav_policy + '_iter_' + str(i+1)
                self.check_point_in(
                    self.__actor_net.state_dict(), agent_name, self._w, None)

            trans = random.sample(self.__memor, self._batch_size)
            stats = []
            actis = []
            rewas = []
            next_stats = []
            for tran in trans:
                state, actio, rewar, next_state = tran
                stats.append(ast.literal_eval(state))
                actis.append(ast.literal_eval(actio))
                rewas.append(ast.literal_eval(rewar))
                next_stats.append(ast.literal_eval(next_state))

            s = torch.tensor(
                stats, dtype=torch.float32).reshape(-1, self._state_size)
            # LongTensor for idx selection
            a = torch.tensor(actis, dtype=torch.float32)
            r = torch.tensor(rewas, dtype=torch.float32)
            n_s = torch.tensor(
                next_stats, dtype=torch.float32).reshape(-1, self._state_size)

            # update two critic networks
            self.__criti_optim.zero_grad()
            self.__criti_optim_dual.zero_grad()
            # current estimate
            s_a = torch.concat((s, a.unsqueeze(dim=1)), dim=1)
            for param in self.__criti_net.parameters():
                param.requires_grad = True
            for param in self.__criti_net_dual.parameters():
                param.requires_grad = True
            Q = self.__criti_net(s_a)
            Q_dual = self.__criti_net_dual(s_a)

            # Bellman backup for Q function
            targe_imagi_a = self.__targe_actor_net(n_s)  # (batch_size, 1)
            targe_imagi_a += torch.clamp(torch.randn_like(targe_imagi_a)
                                         * 0.2, -0.2, 0.2)
            targe_imagi_a = torch.clamp(targe_imagi_a, 0, 1)

            s_targe_imagi_a = torch.concat((n_s, targe_imagi_a), dim=1)
            with torch.no_grad():
                q_polic_targe = self.__targe_criti_net(s_targe_imagi_a)
                q_polic_targe_dual = self.__targe_criti_net_dual(
                    s_targe_imagi_a)
                min_q_polic_targe = torch.minimum(
                    q_polic_targe, q_polic_targe_dual)

                # r is (batch_size, ), need to align with output from NN
                back_up = r.unsqueeze(1) + self._gamma * min_q_polic_targe

            # MSE loss against Bellman backup
            td = Q - back_up
            td_dual = Q_dual - back_up
            criti_loss = (td**2).mean()
            criti_loss_dual = (td_dual**2).mean()
            # update critic parameters
            criti_loss.backward()
            criti_loss_dual.backward()

            self.__criti_optim.step()
            self.__criti_optim_dual.step()
            self.__criti_updat_count += 1

            wandb.log({'criti_loss': criti_loss.item(), 'Q': Q.mean().item(),
                       'criti_loss_dual': criti_loss_dual.item(), 'Q_dual': Q_dual.mean().item()})

            if self.__criti_updat_count % self.__actor_delay_round == 0:
                self.__criti_updat_count = 0

                # update actor network
                self.__actor_optim.zero_grad()
                imagi_a = self.__actor_net(s)
                s_imagi_a = torch.concat((s, imagi_a), dim=1)
                # Freeze Q-network to save computational efforts
                for param in self.__criti_net.parameters():
                    param.requires_grad = False
                Q = self.__criti_net(s_imagi_a)
                actor_loss = -Q.mean()
                actor_loss.backward()
                self.__actor_optim.step()

                # Finally, update target networks by polyak averaging.
                with torch.no_grad():
                    for p, p_targ in zip(self.__actor_net.parameters(), self.__targe_actor_net.parameters()):
                        p_targ.data.mul_(self._polya)
                        p_targ.data.add_((1 - self._polya) * p.data)
                    for p, p_targ in zip(self.__criti_net.parameters(), self.__targe_criti_net.parameters()):
                        p_targ.data.mul_(self._polya)
                        p_targ.data.add_((1 - self._polya) * p.data)
                    for p, p_targ in zip(self.__criti_net_dual.parameters(), self.__targe_criti_net_dual.parameters()):
                        p_targ.data.mul_(self._polya)
                    p_targ.data.add_((1 - self._polya) * p.data)

                wandb.log({'criti_loss': criti_loss.item(),
                           'actor_loss': actor_loss.item()})

        # save actor model
        # os.makedirs(model_dir_path, exist_ok=True)
