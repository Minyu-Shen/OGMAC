import torch
import torch.nn.functional as F
from .approximator import MLP

# the hidden dimension when concatenate continuous and discrete state
hidde_out_size = 8


class Actor_Net(torch.nn.Module):
    def __init__(self, state_size, hidde_size, embed_discr_state_info, init_type='default'):
        '''
            state_size: size of the state (backward and forward spacing, and discrete state if any)
            hidde_size: tuple with the number of neurons in each hidden layer for continuous feature
            embed_discr_state_info: tuple with two elements, the first element is a bool value indicating 
                        whether to embed discrete state, the second element is the size of discrete state
        '''
        super(Actor_Net, self).__init__()
        self.__is_embed_discr_state, self.__discr_size = embed_discr_state_info

        if self.__is_embed_discr_state:
            assert self.__discr_size is not None, 'discr_size must be not None if is_embed_discr_sate is True'
            # the size that state and discrete state will be projected
            self.conti_mlp = MLP(state_size-1, hidde_out_size,
                                 hidde_size, outpu='logits', init_type=init_type)
            self.embed_layer = torch.nn.Embedding(
                self.__discr_size, hidde_out_size)
            self.outpu_layer = torch.nn.Linear(hidde_out_size, 1)
        else:
            self.conti_mlp = MLP(state_size-1, 1, hidde_size,
                                 outpu='sigmoid', init_type=init_type)

    def forward(self, x):
        '''
            x: x[0] and x[1] are continuous states, x[2] is discrete state
        '''
        if type(x) == list:
            x = torch.tensor(x, dtype=torch.float32).reshape(-1, 3)
        if self.__is_embed_discr_state:
            conti_x = x[:, 0:2]
            discr_x = x[:, 2].long()

            hidde_conti_x = self.conti_mlp(conti_x)
            hidde_discr_x = self.embed_layer(discr_x)
            hidde_x = torch.add(hidde_conti_x, hidde_discr_x)
            # hidde_x = torch.cat((hidde_conti_x, hidde_discr_x), dim=1)
            logit = self.outpu_layer(hidde_x)
            return torch.sigmoid(logit)
        else:
            conti_x = x[:, 0:2]
            return self.conti_mlp(conti_x)


class Critic_Net(torch.nn.Module):
    def __init__(self, state_size, hidde_size, embed_discr_sate_info, init_type='default'):
        '''
            state_size: size of the state (backward and forward spacing)
            hidde_size: tuple with the number of neurons in each hidden layer for continuous feature
            discr_state_size: discrete state size (e.g., stop id)
            discr_size: number of discrete states (e.g., the maximum stop id)
        '''
        super(Critic_Net, self).__init__()
        self.__is_embed_discr_state, self.__discr_size = embed_discr_sate_info

        if self.__is_embed_discr_state:
            # input state and action
            self.conti_mlp = MLP(state_size-1+1, hidde_out_size,
                                 hidde_size, outpu='logits', init_type=init_type)
            self.embed_layer = torch.nn.Embedding(
                self.__discr_size, hidde_out_size)
            self.outpu_layer = torch.nn.Linear(hidde_out_size, 1)
        else:
            self.conti_mlp = MLP(state_size-1+1, 1, hidde_size,
                                 outpu='logits', init_type=init_type)

    def forward(self, x):
        '''
            x: x[0] and x[1] are continuous states, x[2] is discrete state, x[3] is action
        '''

        if self.__is_embed_discr_state:
            # if type(x) == list:
            #     x = torch.tensor(x, dtype=torch.float32).reshape(-1, 3)
            conti_x = x[:, 0:2]
            discr_x = x[:, 2].long()
            a = x[:, 3].unsqueeze(1)
            conti_x_a = torch.cat((conti_x, a), dim=1)

            hidde_conti_x_a = self.conti_mlp(conti_x_a)
            hidde_discr_x = self.embed_layer(discr_x)
            hidde_x = torch.add(hidde_conti_x_a, hidde_discr_x)
            # hidde_x = torch.cat((hidde_conti_x_a, hidde_discr_x), dim=1)
            logit = self.outpu_layer(hidde_x)
            return logit
        else:
            conti_x = x[:, 0:2]
            a = x[:, 3].unsqueeze(1)
            conti_x_a = torch.cat((conti_x, a), dim=1)
            logit = self.conti_mlp(conti_x_a)
            return logit


if __name__ == '__main__':
    actor_net = Actor_Net(2, (64,))
    a = actor_net([[0.1, 0.2], [0.3, 0.4]], [9, 5])
    print(a)
    for name, param in actor_net.named_parameters():
        print(name, param.shape)
