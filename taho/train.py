import numpy as np
import torch
from util import TimeRecorder


"""
EpochTrainer for training recurrent models on single sequence of inputs and outputs,
by chunking into bbtt-long segments.
"""



class EpochTrainer(object):
    def __init__(self, model, optimizer, epochs, X, Y, dt, batch_size=1, gpu=False, bptt=50, state0_choice_prob=0.5,
                 short_encoding=False, logging=None, debug=False):


        tr = TimeRecorder()
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.X, self.Y, self.dt = X, Y, dt
        self.batch_size = batch_size
        self.gpu = gpu
        self.Xtrain, self.Ytrain = None, None
        self.train_inds = []
        self.bptt = bptt  # for now: constant segment length, hence constant train indices

        self.all_states = None
        self.rnn_states = None
        self.state0_choice_prob = state0_choice_prob
        self.debug = debug
        self.logging = logging
        self.short_encoding = short_encoding

        self.logging('Trainer initialization')

        with tr('set_train_tensor'):
            self.set_train_tensors()


        self.logging('Initialized epoch trainer: original shape for X', self.X.shape, 'and for Y', self.Y.shape)
        self.logging('and segmented size (segments, bptt, in) for X', self.Xtrain.size(),
              'and (segments, bptt, out) for Y', self.Ytrain.size())
        self.logging('for batch size', self.batch_size)
        self.logging('Time cost {}'.format(tr))

    def set_train_tensors(self):

        tr = TimeRecorder()

        with tr('batch data generation'):
            w = self.bptt
            N = self.X.shape[0]
            Xtrain = np.asarray([self.X[i:min(N, i + w), :] for i in range(max(1, N - w + 1))])  # (instances N-w+1, w, k_in)
            Ytrain = np.asarray([self.Y[i:min(N, i + w), :] for i in range(max(1, N - w + 1))])  # (instances N-w+1, w, k_out)
            dttrain = np.asarray([self.dt[i:min(N, i + w), :] for i in range(max(1, N - w + 1))])  # (instances N-w+1, w, k_out)
            Xtrain = torch.tensor(Xtrain, dtype=torch.float)
            Ytrain = torch.tensor(Ytrain, dtype=torch.float)
            dttrain = torch.tensor(dttrain, dtype=torch.float)

        with tr('batch data to cuda'):
            self.Xtrain = Xtrain.cuda() if self.gpu else Xtrain
            self.Ytrain = Ytrain.cuda() if self.gpu else Ytrain
            self.dttrain = dttrain.cuda() if self.gpu else dttrain
            # self.train_inds = list(range(self.Xtrain.size(0)))  # all instances
            self.train_inds = list(range(self.Xtrain.size(0)-1))  # all instances, minus one to prepare historical seq

        with tr('long seq data generation'):
            Xtrain1 = torch.tensor(self.X, dtype=torch.float).unsqueeze(0)  # (1, seq_len, n_in)  all lengths
            Ytrain1 = torch.tensor(self.Y, dtype=torch.float).unsqueeze(0)  # (1, seq_len, n_out)  all lengths
            dttrain1 = torch.tensor(self.dt, dtype=torch.float).unsqueeze(0)
            self.Xtrain1 = Xtrain1.cuda() if self.gpu else Xtrain1
            self.Ytrain1 = Ytrain1.cuda() if self.gpu else Ytrain1
            self.dttrain1 = dttrain1.cuda() if self.gpu else dttrain1

        self.logging('preparing data: {}'.format(tr))

    def set_states(self):
        with torch.no_grad():  #no backprob beyond initial state for each chunk.
            all_states = self.model(self.Xtrain1, state0=None, dt=self.dttrain1)[1].squeeze(
                0)  # (seq_len, 2)  no state given to model -> start with model.state0
            self.all_states = all_states.data

        self.rnn_states = self.model.generate_state0(torch.cat([self.Xtrain1, self.Ytrain1], dim=-1), last=False)[0]
        # all_states_posterior = self.model.rnn_encoder(self.Ytrain1)

    def __call__(self, epoch):


        np.random.shuffle(self.train_inds)

        # iterations within current epoch
        epoch_loss = 0.
        cum_bs = 0

        # Generating Time Recoder
        tr = TimeRecorder()

        # train initial state only once per epoch

        import time
        t = time.time()
        self.model.train()
        self.model.zero_grad()
        # Y_pred, _ = self.model(self.Xtrain1[:, :self.bptt, :], dt=self.dttrain1[:, :self.bptt, :])
        # nonan_indexes = ~torch.isnan(self.Ytrain1[:, :self.bptt, :])
        # #(no state given: use model.state0)
        # loss = self.model.criterion(Y_pred[nonan_indexes], self.Ytrain1[:, :self.bptt, :][nonan_indexes])
        # loss.backward()
        # self.optimizer.step()


        # set all states only once per epoch (trains much faster than at each iteration)
        # (no gradient through initial state for each chunk)
        with tr('set all states'):
            self.set_states()

        for i in range(int(np.ceil((len(self.train_inds)-1) / self.batch_size))):
            # get indices for next batch
            iter_inds = self.train_inds[i * self.batch_size: (i + 1) * self.batch_size]
            bs = len(iter_inds)

            # get initial states for next batch
            #self.set_states()  #only 1 x per epoch, much faster

            # get batch input and target data
            cum_bs += bs

            pre_X = self.Xtrain[iter_inds, :, :]  # (batch, bptt, k_in)
            pre_Y_target = self.Ytrain[iter_inds, :, :]
            pre_dt = self.dttrain[iter_inds, :, :]  # (batch, bptt, 1)
            iter_inds_next = [x+1 for x in iter_inds]
            X = self.Xtrain[iter_inds_next, :, :]  # (batch, bptt, k_in)
            dt = self.dttrain[iter_inds_next, :, :]  # (batch, bptt, 1)
            Y_target = self.Ytrain[iter_inds_next, :, :]

            with tr('state0'):

                state0 = self.all_states[iter_inds, :]  # (batch, k_state)

                if self.short_encoding:
                    state0_short_encoding = self.model.generate_state0(torch.cat([pre_X, pre_Y_target], dim=-1), dt=pre_dt)
                    state0_choice_rand = torch.rand((bs,))
                    state0[state0_choice_rand < self.state0_choice_prob] = state0_short_encoding[state0_choice_rand < self.state0_choice_prob]

            # training
            self.model.train()
            self.model.zero_grad()

            with tr('forward'):
                Y_pred, _ = self.model(X, state0=state0, dt=dt)

            loss = self.model.criterion(Y_pred, Y_target)
            with tr('backward'):
                loss.backward()

            # debug: observe gradients
            self.optimizer.step()
            epoch_loss += loss.item() * bs
            self.logging('Epoch {}, iters {}-{}, loss {:.4f}, time {}'.format(
                epoch, i, int(np.ceil((len(self.train_inds)-1) / self.batch_size)), float(loss.item()), tr
            ))

            # for i in range(int(np.ceil((len(self.train_inds)-1) / self.batch_size))):
            if self.debug and i>=1:
                break

        epoch_loss /= cum_bs

        return epoch_loss




