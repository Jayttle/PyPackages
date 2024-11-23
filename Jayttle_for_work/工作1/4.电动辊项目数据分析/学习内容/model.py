import torch
import random
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from multiprocessing import Pool
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as pyGDataLoader


# 模型：GAN生成对抗网络异常检测模型
class GANmodel(nn.Module):
    def __init__(self, input_dim, latent_dim=100, hidden_dim=128):
        """
        初始化GAN模型
        :param input_dim: 输入维度
        :param latent_dim: 隐变量维度
        """
        super(GANmodel, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Generator
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),  # 增加隐藏层的维度
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim * 2, hidden_dim),  # 更深的隐藏层
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, input_dim),  # 输出层，输出传感器数据
            nn.Tanh()  # 根据数据范围选择合适的激活函数
        )

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 512),  # 输入层对应传感器数据的维度
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),  # 第一隐藏层
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),  # 第二隐藏层
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),  # 输出层，给出真假判断
            nn.Sigmoid()  # 输出 [0, 1] 的概率
        )

    def forward(self, z):
        """
        前向传播
        :param z: 隐变量
        :return: 生成样本
        """
        return self.generator(z)

    def discriminate(self, x):
        """
        判别器前向传播
        :param x: 输入样本
        :return: 判别概率
        """
        return self.discriminator(x)

    def fit(self, data, epochs=200, batch_size=64, lr=0.0001, beta1=0.5):
        """
        训练GAN模型
        :param data: 训练数据
        :param epochs: 训练轮数
        :param batch_size: 批量大小
        :param lr: 学习率
        :param beta1: Adam优化器参数
        """
        criterion = nn.BCELoss()  # 二元交叉熵损失
        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

        dataset = torch.tensor(data).float()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(epochs):
            for i, inputs in enumerate(dataloader):
                # 训练判别器
                valid_labels = torch.ones(inputs.size(0), 1)  # 真实样本标签为1
                fake_labels = torch.zeros(inputs.size(0), 1)  # 生成样本标签为0

                # 判别真实样本
                optimizer_d.zero_grad()
                real_outputs = self.discriminate(inputs)
                d_loss_real = criterion(real_outputs, valid_labels)

                # 生成假样本
                z = torch.randn(inputs.size(0), self.latent_dim)
                fake_samples = self(z)
                fake_outputs = self.discriminate(fake_samples.detach())
                d_loss_fake = criterion(fake_outputs, fake_labels)

                # 总判别器损失
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_d.step()

                # 训练生成器
                optimizer_g.zero_grad()
                fake_outputs = self.discriminate(fake_samples)
                g_loss = criterion(fake_outputs, valid_labels)
                g_loss.backward()
                optimizer_g.step()

    def anomaly_score(self, data):
        """
        计算异常分数
        :param data: 输入数据
        :param threshold: 阈值
        :return: 异常分数
        """
        self.eval()
        with torch.no_grad():
            inputs = torch.tensor(data).float()
            # 判别器对生成样本的输出作为异常分数
            outputs = self.discriminate(inputs)
            # 异常分数 = 1 - 判别器输出概率
            anomaly_scores = 1 - outputs.squeeze().numpy()
        return anomaly_scores



# 模型：VAE自编码异常检测模型
class VAEmodel(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        """
        初始化VAE自编码模型
        :param input_dim: 输入维度
        :param latent_dim: 隐变量维度
        """
        super(VAEmodel, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim * 2)  # z_mean and z_log_var
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.input_dim),
            nn.Sigmoid()  # 假设输入已经归一化到[0, 1]
        )

    def reparameterize(self, z_mean, z_log_var):
        """
        重参数化技巧
        :param z_mean: 均值
        :param z_log_var: 对数方差
        :return: 重新参数化后的向量
        """
        epsilon = torch.randn_like(z_log_var)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据
        :return: 解码输出, 均值, 对数方差
        """
        # 编码
        encoded = self.encoder(x)
        z_mean, z_log_var = encoded[:, :self.latent_dim], encoded[:, self.latent_dim:]

        # 重参数化技巧
        z = self.reparameterize(z_mean, z_log_var)

        # 解码
        decoded = self.decoder(z)

        return decoded, z_mean, z_log_var

    def fit(self, X, epochs=50, batch_size=32):
        """
        训练VAE模型
        :param X: 训练数据
        :param epochs: 训练轮数
        :param batch_size: 批量大小
        """
        optimizer = optim.Adam(self.parameters())
        criterion = nn.MSELoss()  # 均方误差损失

        for epoch in range(epochs):
            running_loss = 0.0
            for i in range(0, len(X), batch_size):
                inputs = torch.tensor(X[i:i + batch_size]).float()
                optimizer.zero_grad()
                outputs, z_mean, z_log_var = self(inputs)
                loss = criterion(outputs, inputs)  # 重构损失

                # KL散度损失
                kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

                loss += kl_loss
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if epoch % 10 == 9:
                print(f"第 {epoch + 1} 轮, 损失: {running_loss}")

    def anomaly_score(self, X):
        """
        计算异常分数
        :param X: 输入数据
        :return: 异常分数
        """
        self.eval()
        with torch.no_grad():
            inputs = torch.tensor(X).float()
            outputs, _, _ = self(inputs)
            reconstruction_error = torch.mean((inputs - outputs) ** 2, dim=1).numpy()
        return reconstruction_error


class LinearBlock(torch.nn.Module):
    """
    线性层块，支持激活函数、Dropout、跳跃连接和 Batch Ensemble
    """

    def __init__(self, in_channels, out_channels,
                 bias=False, activation='tanh',
                 skip_connection=None, dropout=None, be_size=None):

        super(LinearBlock, self).__init__()

        self.act = activation
        self.skip_connection = skip_connection
        self.dropout = dropout
        self.be_size = be_size

        if activation is not None:
            self.act_layer, _ = torch.nn.Tanh(), torch.tanh

        if dropout is not None:
            self.dropout_layer = torch.nn.Dropout(p=dropout)

        if be_size is not None:
            bias = False
            self.ri = torch.nn.Parameter(torch.randn(be_size, in_channels))
            self.si = torch.nn.Parameter(torch.randn(be_size, out_channels))

        self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        if self.be_size is not None:
            R = torch.repeat_interleave(self.ri, int(x.shape[0] / self.be_size), dim=0)
            S = torch.repeat_interleave(self.si, int(x.shape[0] / self.be_size), dim=0)

            x1 = torch.mul(self.linear(torch.mul(x, R)), S)
        else:
            x1 = self.linear(x)

        if self.act is not None:
            x1 = self.act_layer(x1)

        if self.dropout is not None:
            x1 = self.dropout_layer(x1)

        if self.skip_connection == 'concat':
            x1 = torch.cat([x, x1], axis=1)

        return x1


class MLPnet(torch.nn.Module):
    def __init__(self, n_features, n_hidden=[500, 100], n_emb=20, activation='tanh',
                 skip_connection=None, dropout=None, be_size=None):
        super(MLPnet, self).__init__()
        self.skip_connection = skip_connection
        self.n_emb = n_emb

        assert activation in ['relu', 'tanh', 'sigmoid', 'leaky_relu']

        if type(n_hidden) == int: n_hidden = [n_hidden]
        if type(n_hidden) == str: n_hidden = n_hidden.split(','); n_hidden = [int(a) for a in n_hidden]
        num_layers = len(n_hidden)

        self.be_size = be_size

        self.layers = []
        for i in range(num_layers + 1):
            in_channels, out_channels = self.get_in_out_channels(i, num_layers, n_features,
                                                                 n_hidden, n_emb, skip_connection)
            self.layers += [LinearBlock(in_channels, out_channels,
                                        activation=activation if i != num_layers else None,
                                        skip_connection=skip_connection if i != num_layers else 0,
                                        dropout=dropout,
                                        be_size=be_size)]
        self.network = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        if self.be_size is not None:
            x = x.repeat(self.be_size, 1)
        x = self.network(x)
        return x

    def get_in_out_channels(self, i, num_layers, n_features, n_hidden, n_emb, skip_connection):
        if skip_connection is None:
            in_channels = n_features if i == 0 else n_hidden[i - 1]
            out_channels = n_emb if i == num_layers else n_hidden[i]
        elif skip_connection == 'concat':
            in_channels = n_features if i == 0 else np.sum(n_hidden[:i]) + n_features
            out_channels = n_emb if i == num_layers else n_hidden[i]
        else:
            raise NotImplementedError('')
        return in_channels, out_channels

class DIF:
    def __init__(self, network_name='mlp', network_class=None,
                 n_ensemble=50, n_estimators=6, max_samples=256,
                 hidden_dim=[500, 100], rep_dim=20, skip_connection=None, dropout=None, activation='tanh',
                 data_type='tabular', batch_size=64,
                 new_score_func=True, new_ensemble_method=True,
                 random_state=42, device='cpu', n_processes=1,
                 verbose=0, **network_args):
        # super(DeepIsolationForest, self).__init__(contamination=contamination)

        if data_type not in ['tabular', 'graph', 'ts']:
            raise NotImplementedError('unsupported data type')

        self.data_type = data_type
        self.n_ensemble = n_ensemble
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.batch_size = batch_size

        self.new_score_func = new_score_func
        self.new_ensemble_method = new_ensemble_method

        self.device = device
        self.n_processes = n_processes
        self.verbose = verbose

        self.network_args = network_args
        self.Net = MLPnet
        if network_name == 'mlp':
            self.network_args['n_hidden'] = hidden_dim
            self.network_args['n_emb'] = rep_dim
            self.network_args['skip_connection'] = skip_connection
            self.network_args['dropout'] = dropout
            self.network_args['activation'] = activation
            self.network_args['be_size'] = None if self.new_ensemble_method == False else self.n_ensemble

        if network_class is not None:
            self.Net = network_class
        print(f'network additional parameters: {network_args}')

        self.transfer_flag = True

        self.n_features = -1
        self.net_lst = []
        self.clf_lst = []
        self.x_reduced_lst = []
        self.score_lst = []

        self.set_seed(random_state)
        return

    def fit(self, X, y=None):
        """
        Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        start_time = time.time()
        self.n_features = X.shape[-1] if self.data_type != 'graph' else max(X.num_features, 1)
        ensemble_seeds = np.random.randint(0, 1e+5, self.n_ensemble)

        if self.verbose >= 2:
            net = self.Net(n_features=self.n_features, **self.network_args)
            print(net)

        self._training_transfer(X, ensemble_seeds)

        if self.verbose >= 2:
            it = tqdm(range(self.n_ensemble), desc='clf fitting', ncols=80)
        else:
            it = range(self.n_ensemble)

        for i in it:
            self.clf_lst.append(
                IsolationForest(n_estimators=self.n_estimators,
                                max_samples=self.max_samples,
                                random_state=ensemble_seeds[i])
            )
            self.clf_lst[i].fit(self.x_reduced_lst[i])

        if self.verbose >= 1:
            print(f'training done, time: {time.time() - start_time:.1f}')
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """

        test_reduced_lst = self._inference_transfer(X)
        final_scores = self._inference_scoring(test_reduced_lst, n_processes=self.n_processes)
        return final_scores

    def _training_transfer(self, X, ensemble_seeds):
        if self.new_ensemble_method:
            self.set_seed(ensemble_seeds[0])
            net = self.Net(n_features=self.n_features, **self.network_args).to(self.device)
            self.net_init(net)

            self.x_reduced_lst = self.deep_transfer_batch_ensemble(X, net)
            self.net_lst.append(net)
        else:
            for i in tqdm(range(self.n_ensemble), desc='training ensemble process', ncols=100, leave=None):
                self.set_seed(ensemble_seeds[i])
                net = self.Net(n_features=self.n_features, **self.network_args).to(self.device)
                self.net_init(net)

                self.x_reduced_lst.append(self.deep_transfer(X, net))
                self.net_lst.append(net)
        return

    def _inference_transfer(self, X):
        if self.data_type == 'tabular' and X.shape[0] == self.x_reduced_lst[0].shape[0]:
            return self.x_reduced_lst

        test_reduced_lst = []
        if self.new_ensemble_method:
            test_reduced_lst = self.deep_transfer_batch_ensemble(X, self.net_lst[0])
        else:
            for i in tqdm(range(self.n_ensemble), desc='testing ensemble process', ncols=100, leave=None):
                x_reduced = self.deep_transfer(X, self.net_lst[i])
                test_reduced_lst.append(x_reduced)
        return test_reduced_lst

    def _inference_scoring(self, x_reduced_lst, n_processes):
        if self.new_score_func:
            score_func = self.single_predict
        else:
            score_func = self.single_predict_abla

        n_samples = x_reduced_lst[0].shape[0]
        self.score_lst = np.zeros([self.n_ensemble, n_samples])
        if n_processes == 1:
            for i in range(self.n_ensemble):
                scores = score_func(x_reduced_lst[i], self.clf_lst[i])
                self.score_lst[i] = scores
        else:
            # multiprocessing predict
            start = np.arange(0, self.n_ensemble, np.ceil(self.n_ensemble / n_processes))
            for j in range(int(np.ceil(self.n_ensemble / n_processes))):
                run_id = start + j
                run_id = np.array(np.delete(run_id, np.where(run_id >= self.n_ensemble)), dtype=int)
                if self.verbose >= 1:
                    print('Multi-processing Running ensemble id :', run_id)

                pool = Pool(processes=n_processes)
                process_lst = [pool.apply_async(score_func, args=(x_reduced_lst[i], self.clf_lst[i]))
                               for i in run_id]
                pool.close()
                pool.join()

                for rid, process in zip(run_id, process_lst):
                    self.score_lst[rid] = process.get()

        final_scores = np.average(self.score_lst, axis=0)

        return final_scores

    def deep_transfer(self, X, net):
        x_reduced = []

        with torch.no_grad():
            if self.data_type != 'graph':
                loader = DataLoader(X, batch_size=self.batch_size, drop_last=False, pin_memory=True, shuffle=False)
                for batch_x in loader:
                    batch_x = batch_x.float().to(self.device)
                    batch_x_reduced = net(batch_x)
                    x_reduced.append(batch_x_reduced)
            else:
                loader = pyGDataLoader(X, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False)
                for data in loader:
                    data.to(self.device)
                    x, edge_index, batch = data.x, data.edge_index, data.batch
                    if x is None:
                        x = torch.ones((batch.shape[0], 1)).to(self.device)
                    x, _ = net(x, edge_index, batch)
                    x_reduced.append(x)

        x_reduced = torch.cat(x_reduced).data.cpu().numpy()
        x_reduced = StandardScaler().fit_transform(x_reduced)
        x_reduced = np.tanh(x_reduced)
        return x_reduced

    def deep_transfer_batch_ensemble(self, X, net):
        x_reduced = []

        with torch.no_grad():
            loader = DataLoader(X, batch_size=self.batch_size, drop_last=False, pin_memory=True, shuffle=False)
            for batch_x in loader:
                batch_x = batch_x.float().to(self.device)
                batch_x_reduced = net(batch_x)

                batch_x_reduced = batch_x_reduced.reshape([self.n_ensemble, batch_x.shape[0], -1])
                x_reduced.append(batch_x_reduced)

        x_reduced_lst = [torch.cat([x_reduced[i][j] for i in range(len(x_reduced))]).data.cpu().numpy()
                         for j in range(x_reduced[0].shape[0])]

        for i in range(len(x_reduced_lst)):
            xx = x_reduced_lst[i]
            xx = StandardScaler().fit_transform(xx)
            xx = np.tanh(xx)
            x_reduced_lst[i] = xx

        return x_reduced_lst

    @staticmethod
    def net_init(net):
        for name, param in net.named_parameters():
            if name.endswith('weight'):
                torch.nn.init.normal_(param, mean=0., std=1.)
        return

    @staticmethod
    def set_seed(seed):
        seed = int(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def single_predict_abla(x_reduced, clf):
        scores = clf.decision_function(x_reduced)
        scores = -1 * scores
        return scores

    @staticmethod
    def single_predict(x_reduced, clf):
        scores = _cal_score(x_reduced, clf)
        return scores


def _cal_score(xx, clf):
    depths = np.zeros((xx.shape[0], len(clf.estimators_)))
    depth_sum = np.zeros(xx.shape[0])
    deviations = np.zeros((xx.shape[0], len(clf.estimators_)))
    leaf_samples = np.zeros((xx.shape[0], len(clf.estimators_)))

    for ii, estimator_tree in enumerate(clf.estimators_):
        # estimator_population_ind = sample_without_replacement(n_population=xx.shape[0], n_samples=256,
        #                                                       random_state=estimator_tree.random_state)
        # estimator_population = xx[estimator_population_ind]

        tree = estimator_tree.tree_
        n_node = tree.node_count

        if n_node == 1:
            continue

        # get feature and threshold of each node in the iTree
        # in feature_lst, -2 indicates the leaf node
        feature_lst, threshold_lst = tree.feature.copy(), tree.threshold.copy()

        # compute depth and score
        leaves_index = estimator_tree.apply(xx)
        node_indicator = estimator_tree.decision_path(xx)

        # The number of training samples in each test sample leaf
        n_node_samples = estimator_tree.tree_.n_node_samples

        # node_indicator is a sparse matrix with shape (n_samples, n_nodes), indicating the path of input data samples
        # each layer would result in a non-zero element in this matrix,
        # and then the row-wise summation is the depth of data sample
        n_samples_leaf = estimator_tree.tree_.n_node_samples[leaves_index]
        d = (np.ravel(node_indicator.sum(axis=1)) + _average_path_length(n_samples_leaf) - 1.0)
        depths[:, ii] = d
        depth_sum += d

        # decision path of data matrix XX
        node_indicator = np.array(node_indicator.todense())

        # set a matrix with shape [n_sample, n_node], representing the feature value of each sample on each node
        # set the leaf node as -2
        value_mat = np.array([xx[i][feature_lst] for i in range(xx.shape[0])])
        value_mat[:, np.where(feature_lst == -2)[0]] = -2
        th_mat = np.array([threshold_lst for _ in range(xx.shape[0])])

        mat = np.abs(value_mat - th_mat) * node_indicator

        exist = (mat != 0)
        dev = mat.sum(axis=1) / (exist.sum(axis=1) + 1e-6)
        deviations[:, ii] = dev

    scores = 2 ** (-depth_sum / (len(clf.estimators_) * _average_path_length([clf.max_samples_])))
    deviation = np.mean(deviations, axis=1)
    leaf_sample = (clf.max_samples_ - np.mean(leaf_samples, axis=1)) / clf.max_samples_

    scores = scores * deviation
    return scores


def _average_path_length(n_samples_leaf):
    """
    The average path length in a n_samples iTree, which is equal to
    the average path length of an unsuccessful BST search since the
    latter has the same structure as an isolation tree.
    Parameters
    ----------
    n_samples_leaf : array-like of shape (n_samples,)
        The number of training samples in each test sample leaf, for
        each estimators.

    Returns
    -------
    average_path_length : ndarray of shape (n_samples,)
    """

    n_samples_leaf = check_array(n_samples_leaf, ensure_2d=False)

    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.
    average_path_length[mask_2] = 1.
    average_path_length[not_mask] = (
            2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
            - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)

