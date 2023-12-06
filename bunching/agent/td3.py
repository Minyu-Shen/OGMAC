import numpy as np
import torch
import wandb

from .event import Event_Handler
from .agent import Agent
from .approximator import MLP


class TD3_On(Agent):
    def __init__(self, config, is_eval) -> None:
        super(TD3_On, self).__init__(config)
        self._is_eval = is_eval
        self._state_size = 2
        self._max_hold = 60.0
        self._w = 0.05
        self._gamma = 0.9
        # actor update is delayed for some rounds
        self.__actor_delay_round = 5

        # a list of episode num that need to save model
        self.__check_poins = [50, 100, 150, 200]

        self.__actor_net = MLP(self._state_size, 1,
                               hidde_size=(64,), outpu='sigmoid')
        self.__criti_net = MLP(self._state_size+1, 1,
                               hidde_size=(64,), outpu='logits')
        self.__criti_net_dual = MLP(
            self._state_size+1, 1, hidde_size=(64,), outpu='logits')

        self.__event_handl = Event_Handler(config, is_rewar_globa=False, is_rewar_track_globa=True,
                                           is_state_globa=False, max_hold=self._max_hold, w=self._w, is_off_policy=True)

        # return to main function (wandb) for recording
        self.hyper_paras = {'agent_name': 'TD3', 'gamma': self._gamma,
                            'w': self._w, 'is_eval': self._is_eval}

        if not self._is_eval:
            self._polya = 0.995
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
            self.__actor_optim = torch.optim.Adam(
                self.__actor_net.parameters(), lr=5e-3)
            self.__criti_optim = torch.optim.Adam(
                self.__criti_net.parameters(), lr=5e-3)
            self.__criti_optim_dual = torch.optim.Adam(
                self.__criti_net_dual.parameters(), lr=5e-3)

            # maintain update counting
            self.__add_event_count = 0
            # counting for critic update
            self.__criti_updat_count = 0
            # update every self.__update_cycle counts
            self.__updat_cycle = 1
            # maintain noise maker
            self._init_noise_level = 0.15
            self.__noise_level = self._init_noise_level
            self._decay_rate = 0.98
        else:
            self.load_model()

    def load_model(self):
        model = self.check_point_out(self.__str__(), self._w, 99)
        self.__actor_net.load_state_dict(model)

    def reset(self, episode, is_record_wandb=False, is_record_transition=False):
        # write to wandb
        super().reset(episode, is_record_wandb)

        if not self._is_eval:
            self.__event_handl.clear_events()
            self.__noise_level = self._decay_rate ** episode * self._init_noise_level
            print(self.__noise_level)

        if (episode+1) in self.__check_poins:
            self.check_point_in(self.__actor_net.state_dict(),
                                self.__str__(), self._w, episode+1)

    def load_model(self):
        self.__actor_net.load_state_dict(
            torch.load(self.get_actor_model_path()))

    def set_wandb_watcher(self):
        wandb.watch(self.__criti_net, log='all', log_freq=10)

    def __str__(self) -> str:
        return 'TD3_ON'

    def cal_hold_time(self, snapshot):
        # actio = self.infer(snapshot.globa_relat_state)
        actio = self.infer(snapshot.local_state).item()
        hold_time = actio * self._max_hold
        # record departure event and log reward
        track_equal_rewar, track_inten_rewar, track_rewar = self.__event_handl.add_event(
            snapshot, actio, hold_time, self._w)
        self.track(track_rewar, track_equal_rewar,
                   track_inten_rewar, hold_time)
        if not self._is_eval:
            self.__add_event_count += 1
            # if accumulated event num is enough, push it to buffer
            if self.__event_handl.get_trans_num_by_bus() >= self._batch_size:
                self.__event_handl.push_transition_to_buffer()
            # update policy network parameters
            self.learn()

        return hold_time

    def infer(self, state):
        with torch.no_grad():
            a = self.__actor_net(state)
            # when training, add noise
            if not self._is_eval:
                noise = np.random.normal(0, self.__noise_level)
                a = (a + noise).clip(0, 1)
        return a

    def learn(self):
        if self.__add_event_count % self.__updat_cycle != 0 or self.__event_handl.get_buffer_size() < self._batch_size:
            return

        stats, actis, rewas, next_stats = self.__event_handl.sample_transition(
            self._batch_size)
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
                                     * 0.1, -0.2, 0.2)
        targe_imagi_a = torch.clamp(targe_imagi_a, 0, 1)

        s_targe_imagi_a = torch.concat((n_s, targe_imagi_a), dim=1)
        with torch.no_grad():
            q_polic_targe = self.__targe_criti_net(s_targe_imagi_a)
            q_polic_targe_dual = self.__targe_criti_net_dual(s_targe_imagi_a)
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

        if self.__criti_updat_count % self.__actor_delay_round == 0:
            print(self.__criti_updat_count)
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
