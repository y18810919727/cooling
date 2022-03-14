import numpy as np
import torch
from util import TimeRecorder
import random
from random import shuffle
import os


"""
EpochTrainer for training recurrent models on single sequence of inputs and outputs,
by chunking into bbtt-long segments.
"""

class EpochTrainer(object):
    #前期的准备 np转cuda
    def __init__(self, model, optimizer, epochs, X, Y, states, dt, batch_size=1, gpu=False, bptt=50,all_sqe_nums=False,
                 save_dir='./', short_encoding=False, logging=None, debug=False,mymodel=None):

        tr = TimeRecorder()
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.X, self.Y, self.states, self.dt = X, Y, states, dt
        self.batch_size = batch_size
        self.gpu = gpu
        self.Xtrain, self.Ytrain = None, None
        self.all_sqe_nums = all_sqe_nums

        self.bptt = bptt  # for now: constant segment length, hence constant train indices
        self.w = min(bptt, X.shape[0]//2)
        self.class_criterion = torch.nn.CrossEntropyLoss()

        self.all_states = None
        self.rnn_states = None
        self.debug = debug
        self.logging = logging
        self.short_encoding = short_encoding
        self.save_dir = save_dir
        self.mymodel = mymodel
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
            w = self.w
            Xtrain = []
            Ytrain = []
            dttrain = []
            states_train = []
            Xtrain = np.array(Xtrain)
            for key in self.all_sqe_nums:
                #N = self.X.shape[0]
                head = self.all_sqe_nums[key][0]
                tail = self.all_sqe_nums[key][1]
                N = tail-head

                self.all_sqe_nums[key][0] = Xtrain.shape[0]
                Xtrain =  np.append(Xtrain,np.asarray([self.X[i+head:head+min(N, i + w), :] for i in range(max(1, N - w + 1))])).reshape(-1,800,2) # (instances N-w+1, w, k_in)    取数组中第i个的全部数据
                Ytrain =  np.append(Ytrain,np.asarray([self.Y[i+head:head+min(N, i + w), :] for i in range(max(1, N - w + 1))])).reshape(-1,800,3)  # (instances N-w+1, w, k_out)
                dttrain = np.append(dttrain,np.asarray([self.dt[i+head:head+min(N, i + w), :] for i in range(max(1, N - w + 1))])).reshape(-1,800,1)   # (instances N-w+1, w, k_out)
                states_train = np.append(states_train,np.asarray([self.states[i+head:head+min(N, i + w), :] for i in range(max(1, N - w + 1))])).reshape(-1,800,1)  # (instances N-w+1, w, k_out)
                self.all_sqe_nums[key][1] = Xtrain.shape[0]-self.w

            Xtrain = torch.tensor(Xtrain, dtype=torch.float)
            Ytrain = torch.tensor(Ytrain, dtype=torch.float)
            dttrain = torch.tensor(dttrain, dtype=torch.float)
            states_train = torch.tensor(states_train, dtype=torch.long)

        self.train_list = self.random_list();
        with tr('batch data to cuda'):
            self.Xtrain = Xtrain.cuda() if self.gpu else Xtrain
            self.Ytrain = Ytrain.cuda() if self.gpu else Ytrain
            self.dttrain = dttrain.cuda() if self.gpu else dttrain
            self.states_train = states_train.cuda() if self.gpu else states_train
            # self.train_inds = list(range(self.Xtrain.size(0)))  # all instances
            self.train_inds = list(range(self.Xtrain.size(0)-w))  # all instances, minus w to prepare historical seq

        with tr('long seq data generation'):
            Xtrain1 = torch.tensor(self.X, dtype=torch.float).unsqueeze(0)  # (1, seq_len, n_in)  all lengths
            Ytrain1 = torch.tensor(self.Y, dtype=torch.float).unsqueeze(0)  # (1, seq_len, n_out)  all lengths
            dttrain1 = torch.tensor(self.dt, dtype=torch.float).unsqueeze(0)
            states_train1 = torch.tensor(self.states, dtype=torch.long).unsqueeze(0)
            self.Xtrain1 = Xtrain1.cuda() if self.gpu else Xtrain1
            self.Ytrain1 = Ytrain1.cuda() if self.gpu else Ytrain1
            self.dttrain1 = dttrain1.cuda() if self.gpu else dttrain1
            self.states_train1 = states_train1.cuda() if self.gpu else self.states_train1

        self.logging('preparing data: {}'.format(tr))

    def set_states(self):
        with torch.no_grad():  #no backprob beyond initial state for each chunk.
            all_states = self.model(self.Xtrain1, state0=None, dfa_states=self.states_train1, dt=self.dttrain1)[1].squeeze(
                0)  # (seq_len, 2)  no state given to model -> start with model.state0
            self.all_states = all_states.data

        # all_states_posterior = self.model.rnn_encoder(self.Ytrain1)
    # 选出应该打乱的序列
    def random_list(self):
        train_list=[]
        for key in self.all_sqe_nums:
            a = [i for i in range(self.all_sqe_nums[key][0],(self.all_sqe_nums[key][1]))]
            print(a[-1])
            train_list += a
        return train_list

    # #字典随机排序，，使其能够随机训练某个训练集
    # def random_dic(self,dicts):
    #     dict_key_ls = list(dicts.keys())
    #     random.shuffle(dict_key_ls)
    #     new_dic = {}
    #     for key in dict_key_ls:
    #         new_dic[key] = dicts.get(key)
    #     return new_dic

    def __call__(self, epoch) -> object:

        #sqe_head = 0;
        cum_bs = 0
        epoch_loss = 0.
        np.random.shuffle(self.train_list)  # 每个下标对应一个窗口，打散
        train_inds = self.train_list

        # Generating Time Recoder
        tr = TimeRecorder()

        # train initial state only once per epoch

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
        # with tr('set all states'):
        #     self.set_states()
        dfa_states_classifications_pred_all = []
        dfa_states_classifications_label_all = []

        for i in range(int(np.ceil(len(train_inds) / self.batch_size))):  #每次训练batch_size次窗口
            # get indices for next batch
            iter_inds = train_inds[i * self.batch_size: (i + 1) * self.batch_size]# 提取最开始2000个batch_size
            iter_inds_next = [x+self.w for x in iter_inds]  #把
            bs = len(iter_inds)

            # get initial states for next batch
            #self.set_states()  #only 1 x per epoch, much faster

            # get batch input and target data
            cum_bs += bs

            pre_X = self.Xtrain[iter_inds, :, :]  # (batch, bptt, k_in)  #过去的xy
            pre_dt = self.dttrain[iter_inds, :, :]  # (batch, bptt, 1)
            pre_Y_target = self.Ytrain[iter_inds, :, :]
            pre_s = self.states_train[iter_inds, :, :]

            X = self.Xtrain[iter_inds_next, :, :]  # (batch, bptt, k_in)   #未来的xy
            dt = self.dttrain[iter_inds_next, :, :]  # (batch, bptt, 1)
            Y_target = self.Ytrain[iter_inds_next, :, :]
            s = self.states_train[iter_inds_next]

            # training

            self.model.train()
            self.model.zero_grad()

            pre_outputs, pre_states = self.model.forward_posterior(pre_X,torch.cat([pre_X, pre_Y_target], dim=-1),dfa_states=pre_s, dt=pre_dt)
            state0 = pre_states[:, -1]    #得到初始状态# (4000,800)

            with tr('forward'):
                Y_pred, states_pred = self.model.forward_prediction(X, state0=state0, dfa_states=s, dt=dt)

            ht = state0[:,3:-2]
            # Y_pred_power = ((Y_pred[:, :, 2]*dt.reshape(-1,800)).sum(axis=1))
            # Y_target_power = ((Y_target[:, :, 2]*dt.reshape(-1,800)).sum(axis=1))


            loss_h = torch.norm(ht,p = 2,dim =1).sum()/4000


            loss_pred = self.model.criterion(
                torch.cat([pre_outputs, Y_pred], dim=1),#
                torch.cat([pre_Y_target, Y_target], dim=1)
            )

            #
            # loss_power = self.model.criterion(
            #     Y_pred_power,#只考虑预测
            #     Y_target_power
            # )


            if self.mymodel != 'one':
                # The model determines the dfa states at time t by itself based on h(t-1), state(t-1), x(t).
                #时间预测状态
                states_last_step = torch.cat([state0.unsqueeze(dim=1), states_pred[:,:-1, :]], dim=1)
                yt_ht_for_state_transform = states_last_step[..., :-2]
                cum_t_for_state_transform = states_last_step[..., -2:-1] + dt
                s_for_state_transform = states_last_step[..., -1:]

                s_cum_t_for_state_transform = torch.sigmoid(cum_t_for_state_transform)
                all_dfa_states_tag, all_dfa_states_prob, extra_info = self.model.states_classification(
                    torch.cat([
                        yt_ht_for_state_transform,
                        cum_t_for_state_transform,
                        s_for_state_transform
                    ], dim=-1),
                    inputs=X,t=s_cum_t_for_state_transform
                )

                states_label = s
                if 'predicted_stop_cum_time' in extra_info.keys():
                    stop_time_label = torch.zeros_like(extra_info['real_cum_time']) * float('nan')
                    stop_time_label_tmp = torch.zeros_like(stop_time_label[:,0]) * float('nan') # make a nan Tensor with shape (bs, 1)
                    for l in reversed(range(stop_time_label.size()[1])):
                        updated_place = (states_label[:, l] != s_for_state_transform[:, l]).squeeze(dim=1)
                        if torch.any(updated_place):
                            stop_time_label_tmp[updated_place] = extra_info['real_cum_time'][updated_place, l]
                            stop_time_label[:, l] = stop_time_label_tmp.detach().clone()

                    not_null_indices = ~torch.logical_or(
                        torch.isnan(stop_time_label),
                        torch.isnan(extra_info['predicted_stop_cum_time']),
                    )
                    loss_classification = torch.nn.functional.mse_loss(
                        stop_time_label[not_null_indices],
                        extra_info['predicted_stop_cum_time'][not_null_indices]
                    )
                    # for state_index in range(6):
                    #     print('state : %d, mean of label %.2f, mean of prediction %.2f' % (state_index, float(
                    #         stop_time_label[not_null_indices][s_for_state_transform[not_null_indices] == state_index].mean()
                    #     ), float(
                    #         extra_info['predicted_stop_cum_time'][not_null_indices][
                    #             s_for_state_transform[not_null_indices] == state_index
                    #         ].mean()
                    #     )))

                else:
                    #自动状态转换相关
                    loss_classification = self.class_criterion(
                        all_dfa_states_prob.reshape(-1, all_dfa_states_prob.shape[-1]).contiguous(),
                        states_label.reshape(-1).contiguous(),
                    )

                dfa_states_classifications_pred_all.append(all_dfa_states_tag.reshape(-1).detach().cpu())
                dfa_states_classifications_label_all.append(states_label.reshape(-1).detach().cpu())
            if self.mymodel != 'one':
                loss = loss_classification + loss_pred+loss_h
            else:
                loss = loss_pred+loss_h

            with tr('backward'):
                loss.backward() #反向传播

            # debug: observe gradients
            self.optimizer.step()

            epoch_loss += loss.item() * bs #梯度下降
            #epoch_loss_pred += loss_pred.item() * bs
           # epoch_loss_power += loss_power.item() *bs
            if self.mymodel == 'one':
                self.logging('Epoch {}, iters {}-{}, train_size {} ,loss {:.4f}, loss_pred {:.4f},time {}'.format(
                    epoch, i+1, int(np.ceil((len(train_inds)-1) / self.batch_size)), bs , float(loss.item()),
                    float(loss_pred.item()), tr
                ))
            else:
                self.logging('Epoch {}, iters {}-{}, train_size {} ,loss {:.4f} loss_h {:.4f},loss_pred {:.4f}, loss_class {:.4f},time {}'.format(
                    epoch, i+1, int(np.ceil((len(train_inds)-1) / self.batch_size)), bs , float(loss.item()),
                    float(loss_h),float(loss_pred.item()), float(loss_classification.item()), tr
                ))
            if self.debug and i>=1:
                break
        epoch_loss /= cum_bs

        return epoch_loss




