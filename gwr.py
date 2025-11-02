from __future__ import division

import logging

import copy
import numpy as np
import networkx as nx
import scipy.spatial.distance as sp
import matplotlib.pyplot as plt






class gwr():

    '''
    Growing When Required (GWR) Neural Gas, after [1]. Constitutes the base class
    for the Online Semi-supervised (OSS) GWR.

    [1] Parisi, G. I., Tani, J., Weber, C., & Wermter, S. (2017).
    Emergence of multimodal action representations from neural network
    self-organization. Cognitive Systems Research, 43, 208-221.
    '''

    def __init__(self, act_thr = 0.35, fir_thr = 0.1, eps_b = 0.1,
                 eps_n = 0.01, tau_b = 0.3, tau_n = 0.1, kappa = 1.05,
                 lab_thr = 0.5, max_age = 100, max_size = 100,
                 random_state = None):
        self.act_thr  = act_thr
        self.fir_thr  = fir_thr
        self.eps_b    = eps_b
        self.eps_n    = eps_n
        self.tau_b    = tau_b
        self.tau_n    = tau_n
        self.kappa    = kappa
        self.lab_thr  = lab_thr
        self.max_age  = max_age
        self.max_size = max_size
        if random_state is not None:
            np.random.seed(random_state)

    def _initialize(self, X, Y):

        logging.info('Initializing the neural gas.')
        self.G = nx.Graph()

        draw = np.random.choice(X.shape[0], size=2, replace=False)


        self.G.add_nodes_from([(0,{'pos' : X[draw[0],:],'fir' : 1, 'label' : Y[draw[0]]})])
        self.G.add_nodes_from([(1,{'pos' : X[draw[1],:],'fir' : 1, 'label' : Y[draw[1]]})])




    def get_positions(self):
        pos = np.array(list(nx.get_node_attributes(self.G, 'pos').values()))
        return pos

    def _get_best_matching(self, x):
        pos = self.get_positions()














        dist = sp.cdist(x, pos, metric='euclidean')
        sorted_dist = np.argsort(dist)

        b = sorted_dist[0,0]
        s = sorted_dist[0,1]
        return b, s


    def _get_activation(self, x, b):
        p = self.G.nodes[b]['pos'][np.newaxis,:]
        dist = sp.cdist(x, p, metric='euclidean')[0,0]
        act = np.exp(-dist)
        return act


    def _make_link(self, b, s):
        self.G.add_edge(b,s,age = 0)



    def _add_node(self, x, y, b, s):
        r = max(self.G.nodes()) + 1
        pos_r = 0.5 * (x + self.G.nodes[b]['pos'])[0,:]
        self.G.add_nodes_from([(r, {'pos' : pos_r, 'fir' : 1, 'label' : y})])
        self.G.remove_edge(b,s)
        self.G.add_edge(r, b, age = 0)
        self.G.add_edge(r, s, age = 0)
        return r


    def _update_network(self, x, b):
        dpos_b = self.eps_b * self.G.nodes[b]['fir']*(x - self.G.nodes[b]['pos'])
        self.G.nodes[b]['pos'] = self.G.nodes[b]['pos'] + dpos_b[0,:]

        neighbors = self.G.neighbors(b)
        for n in neighbors:

            dpos_n = self.eps_n * self.G.nodes[n]['fir'] * (
                     x - self.G.nodes[n]['pos'])
            self.G.nodes[n]['pos'] = self.G.nodes[n]['pos'] + dpos_n[0,:]



            self.G.edges[b,n]['age'] += 1


    def _update_firing(self, b):
        dfir_b = self.tau_b * self.kappa*(1-self.G.nodes[b]['fir']) - self.tau_b
        self.G.nodes[b]['fir'] = self.G.nodes[b]['fir']  + dfir_b


        self.G.nodes[b]['fir'] = np.clip(self.G.nodes[b]['fir'],0,1)



        neighbors = self.G.neighbors(b)
        for n in neighbors:
            dfir_n = self.tau_n * self.kappa * \
                     (1-self.G.nodes[b]['fir']) - self.tau_n
            self.G.nodes[n]['fir'] = self.G.nodes[n]['fir'] + dfir_n
            self.G.nodes[n]['fir'] = np.clip(self.G.nodes[n]['fir'],0,1)


    def _remove_old_edges(self):
        for e in self.G.edges():
            if self.G[e[0]][e[1]]['age'] > self.max_age:
                self.G.remove_edge(*e)
        for node in self.G.nodes():
            if len(self.G.edges(node)) == 0:
                logging.debug('Removing node %s', str(node))
                self.G.remove_node(node)

    def _check_stopping_criterion(self):

        pass

    def _training_step(self, x, y):

        b, s = self._get_best_matching(x)

        self._make_link(b, s)
        act = self._get_activation(x, b)
        fir = self.G.nodes[b]['fir']
        logging.debug('Training step - best matching: %s, %s \n'
                      'Network activation: %s \n'
                      'Firing: %s', str(b), str(s), str(np.round(act,3)),
                      str(np.round(fir,3)))
        if act < self.act_thr and fir < self.fir_thr \
            and len(self.G.nodes()) < self.max_size:
            r = self._add_node(x, y, b, s)
            logging.debug('GENERATE NODE %s', self.G.nodes[r])
        else:
            self._update_network(x, b)
        self._update_firing(b)
        self._remove_old_edges()


    def train(self, X, Y, n_epochs=20, warm_start = False):
        if not warm_start:
            self._initialize(X,Y)
        for n in range(n_epochs):
            print('epoch: ',n)
            logging.info('>>> Training epoch %s', str(n))
            for i in range(X.shape[0]):
                x = X[i,np.newaxis]
                y = Y[i]
                self._training_step(x,y)
                self._check_stopping_criterion()


        logging.info('Training ended - Network size: %s', len(self.G.nodes()))      
        return self.G

    def test(self, X, Y):
        num_correct = 0
        class_by_class = np.zeros(10)
        class_by_class_pred = np.zeros(10)
        for i in range(X.shape[0]):
            x = X[i,np.newaxis]
            y = Y[i]

            class_by_class[int(y)] += 1
            b,s = self._get_best_matching(x)
            act = self._get_activation(x, b)

            y_pred = self.G.nodes[b]['label']
            if (y_pred == y):
                num_correct += 1
                class_by_class_pred[int(y)] += 1
        return num_correct/len(Y),class_by_class_pred/class_by_class

    def _test_best_matching(self, x):
        pos = self.get_positions()


        dist = sp.cdist(x, pos, metric='euclidean')
        sorted_dist = np.argsort(dist)



        return sorted_dist[0,:10]

    def KNearest_test(self, X, Y):
        num_correct = 0
        class_by_class = np.zeros(10)
        class_by_class_pred = np.zeros(10)
        for i in range(X.shape[0]):
            x = X[i,np.newaxis]
            y = Y[i]
            class_by_class[int(y)] += 1
            best_matches = self._test_best_matching(x)
            votes = np.zeros(10)
            for j in best_matches:
                votes[int(self.G.nodes[j]['label'])] += 1
            y_pred = np.argsort(votes)[-1]



            if (y_pred == y):
                num_correct += 1
                class_by_class_pred[int(y)] += 1
        return num_correct/len(Y),class_by_class_pred/class_by_class

    def choose_task(self, X, num_tasks):
        pred_class = np.zeros(len(X))
        for i in range(X.shape[0]):
            x = X[i,np.newaxis]

            best_matches = self._test_best_matching(x)
            votes = np.zeros(num_tasks)
            for j in best_matches:
                votes[int(self.G.nodes[j]['label'])] += 1
            y_pred = np.argsort(votes)[-1]
            pred_class[i] = y_pred
        return pred_class


class gwr2():

    '''
    Growing When Required (GWR) Neural Gas, after [1]. Constitutes the base class
    for the Online Semi-supervised (OSS) GWR.

    [1] Parisi, G. I., Tani, J., Weber, C., & Wermter, S. (2017).
    Emergence of multimodal action representations from neural network
    self-organization. Cognitive Systems Research, 43, 208-221.
    '''

    def __init__(self, act_thr = 0.35, fir_thr = 0.1, eps_b = 0.1,
                 eps_n = 0.01, tau_b = 0.3, tau_n = 0.1, kappa = 1.05,
                 lab_thr = 0.5, max_age = 100, max_size = 100,
                 random_state = None):
        self.act_thr  = act_thr
        self.fir_thr  = fir_thr
        self.eps_b    = eps_b
        self.eps_n    = eps_n
        self.tau_b    = tau_b
        self.tau_n    = tau_n
        self.kappa    = kappa
        self.lab_thr  = lab_thr
        self.max_age  = max_age
        self.max_size = max_size
        self.num_changes = 0
        if random_state is not None:
            np.random.seed(random_state)

    def _initialize(self, X, Y):

        logging.info('Initializing the neural gas.')
        self.G = nx.Graph()

        draw = np.random.choice(X.shape[0], size=2, replace=False)


        self.G.add_nodes_from([(0,{'pos' : X[draw[0],:],'fir' : 1, 'label' : Y[draw[0]], 'best_act' : 1})])
        self.G.add_nodes_from([(1,{'pos' : X[draw[1],:],'fir' : 1, 'label' : Y[draw[1]], 'best_act' : 1})])




    def get_positions(self):
        pos = np.array(list(nx.get_node_attributes(self.G, 'pos').values()))
        return pos

    def _get_best_matching(self, x):
        pos = self.get_positions()


        dist = sp.cdist(x, pos, metric='euclidean')
        sorted_dist = np.argsort(dist)

        b = sorted_dist[0,0]
        s = sorted_dist[0,1]
        return b, s


    def _get_activation(self, x, b):
        p = self.G.nodes[b]['pos'][np.newaxis,:]
        dist = sp.cdist(x, p, metric='euclidean')[0,0]
        act = np.exp(-dist)
        return act


    def _make_link(self, b, s):
        self.G.add_edge(b,s,age = 0)



    def _add_node(self, x, y, b, s):
        r = max(self.G.nodes()) + 1
        pos_r = 0.5 * (x + self.G.nodes[b]['pos'])
        dist = sp.cdist(x, pos_r, metric='euclidean')[0,0]
        act = np.exp(-dist)
        pos_r = pos_r[0,:]
        self.G.add_nodes_from([(r, {'pos' : pos_r, 'fir' : 1, 'label' : y,'best_act' : act})])
        self.G.remove_edge(b,s)
        self.G.add_edge(r, b, age = 0)
        self.G.add_edge(r, s, age = 0)
        return r


    def _update_network(self, x, b):
        dpos_b = self.eps_b * self.G.nodes[b]['fir']*(x - self.G.nodes[b]['pos'])
        self.G.nodes[b]['pos'] = self.G.nodes[b]['pos'] + dpos_b[0,:]

        neighbors = self.G.neighbors(b)
        for n in neighbors:

            dpos_n = self.eps_n * self.G.nodes[n]['fir'] * (
                     x - self.G.nodes[n]['pos'])
            self.G.nodes[n]['pos'] = self.G.nodes[n]['pos'] + dpos_n[0,:]



            self.G.edges[b,n]['age'] += 1


    def _update_firing(self, b):
        dfir_b = self.tau_b * self.kappa*(1-self.G.nodes[b]['fir']) - self.tau_b
        self.G.nodes[b]['fir'] = self.G.nodes[b]['fir']  + dfir_b


        self.G.nodes[b]['fir'] = np.clip(self.G.nodes[b]['fir'],0,1)



        neighbors = self.G.neighbors(b)
        for n in neighbors:
            dfir_n = self.tau_n * self.kappa * \
                     (1-self.G.nodes[b]['fir']) - self.tau_n
            self.G.nodes[n]['fir'] = self.G.nodes[n]['fir'] + dfir_n
            self.G.nodes[n]['fir'] = np.clip(self.G.nodes[n]['fir'],0,1)


    def _remove_old_edges(self):
        for e in self.G.edges():
            if self.G[e[0]][e[1]]['age'] > self.max_age:
                self.G.remove_edge(*e)
        self.G.remove_nodes_from(list(nx.isolates(self.G)))





    def _check_stopping_criterion(self):

        pass

    def _training_step(self, x, y):

        b, s = self._get_best_matching(x)

        self._make_link(b, s)
        act = self._get_activation(x, b)
        fir = self.G.nodes[b]['fir']
        logging.debug('Training step - best matching: %s, %s \n'
                      'Network activation: %s \n'
                      'Firing: %s', str(b), str(s), str(np.round(act,3)),
                      str(np.round(fir,3)))
        if act < self.act_thr and fir < self.fir_thr \
            and len(self.G.nodes()) < self.max_size:
            r = self._add_node(x, y, b, s)
            logging.debug('GENERATE NODE %s', self.G.nodes[r])
        else:
            self._update_network(x, b)
            if ((act > self.G.nodes[b]['best_act']) and (int(self.G.nodes[b]['label']) != int(y))):
                self.G.nodes[b]['label'] = y
                self.num_changes += 1

        self._update_firing(b)
        self._remove_old_edges()


    def train(self, X, Y, n_epochs=20, warm_start = False):
        if not warm_start:
            self._initialize(X,Y)
        for n in range(n_epochs):
            print('epoch: ',n)
            logging.info('>>> Training epoch %s', str(n))
            for i in range(X.shape[0]):
                x = X[i,np.newaxis]
                y = Y[i]
                self._training_step(x,y)
                self._check_stopping_criterion()


        logging.info('Training ended - Network size: %s', len(self.G.nodes())) 
        print('num_changes: ', self.num_changes) 
        self.num_changes = 0    
        return self.G

    def test(self, X, Y):
        num_correct = 0
        class_by_class = np.zeros(10)
        class_by_class_pred = np.zeros(10)
        for i in range(X.shape[0]):
            x = X[i,np.newaxis]
            y = Y[i]
            class_by_class[int(y)] += 1
            b,s = self._get_best_matching(x)
            act = self._get_activation(x, b)

            y_pred = self.G.nodes[b]['label']
            if (y_pred == y):
                num_correct += 1
                class_by_class_pred[int(y)] += 1
        return num_correct/len(Y),class_by_class_pred/class_by_class

    def _test_best_matching(self, x):
        pos = self.get_positions()


        dist = sp.cdist(x, pos, metric='euclidean')
        sorted_dist = np.argsort(dist)



        return sorted_dist[0,:10]

    def KNearest_test(self, X, Y):
        num_correct = 0
        class_by_class = np.zeros(10)
        class_by_class_pred = np.zeros(10)
        for i in range(X.shape[0]):
            x = X[i,np.newaxis]
            y = Y[i]
            class_by_class[int(y)] += 1
            best_matches = self._test_best_matching(x)
            votes = np.zeros(10)
            for j in best_matches:
                votes[int(self.G.nodes[j]['label'])] += 1
            y_pred = np.argsort(votes)[-1]
            if (y_pred == y):
                num_correct += 1
                class_by_class_pred[int(y)] += 1
        return num_correct/len(Y),class_by_class_pred/class_by_class

    def choose_task(self, X, num_tasks):
        pred_class = np.zeros(len(X))
        for i in range(X.shape[0]):
            x = X[i,np.newaxis]

            best_matches = self._test_best_matching(x)
            votes = np.zeros(num_tasks)
            for j in best_matches:
                votes[int(self.G.nodes[j]['label'])] += 1
            y_pred = np.argsort(votes)[-1]
            pred_class[i] = y_pred
        return pred_class



class gwr3():

    '''
    Growing When Required (GWR) Neural Gas, after [1]. Constitutes the base class
    for the Online Semi-supervised (OSS) GWR.

    [1] Parisi, G. I., Tani, J., Weber, C., & Wermter, S. (2017).
    Emergence of multimodal action representations from neural network
    self-organization. Cognitive Systems Research, 43, 208-221.
    '''

    def __init__(self, act_thr = 0.35, fir_thr = 0.1, eps_b = 0.1,
                 eps_n = 0.01, tau_b = 0.3, tau_n = 0.1, alpha_b = 1.05,
                 alpha_n = 1.05, h_0 = 1, sti_s = 1,
                 lab_thr = 0.5, max_age = 100, max_size = 100,
                 random_state = None):
        self.act_thr  = act_thr
        self.fir_thr  = fir_thr
        self.eps_b    = eps_b
        self.eps_n    = eps_n
        self.tau_b    = tau_b
        self.tau_n    = tau_n
        self.alpha_b    = alpha_b
        self.alpha_n    = alpha_n
        self.h_0   = h_0
        self.sti_s   = sti_s
        self.lab_thr  = lab_thr
        self.max_age  = max_age
        self.max_size = max_size
        self.num_changes = 0
        if random_state is not None:
            np.random.seed(random_state)

    def _initialize(self, X, Y):

        logging.info('Initializing the neural gas.')
        self.G = nx.Graph()

        draw = np.random.choice(X.shape[0], size=2, replace=False)


        self.G.add_nodes_from([(0,{'pos' : X[draw[0],:],'fir' : self.sti_s, 'n_best' : 0, 'label' : Y[draw[0]], 'best_act' : 1})])
        self.G.add_nodes_from([(1,{'pos' : X[draw[1],:],'fir' : self.sti_s, 'n_best' : 0, 'label' : Y[draw[1]], 'best_act' : 1})])




    def get_positions(self):
        pos = np.array(list(nx.get_node_attributes(self.G, 'pos').values()))
        return pos

    def _get_best_matching(self, x):
        pos = self.get_positions()


        dist = sp.cdist(x, pos, metric='euclidean')
        sorted_dist = np.argsort(dist)

        b = sorted_dist[0,0]
        s = sorted_dist[0,1]

        self.G.nodes[b]['n_best'] += 1

        return b, s


    def _get_activation(self, x, b):
        p = self.G.nodes[b]['pos'][np.newaxis,:]
        dist = sp.cdist(x, p, metric='euclidean')[0,0]



        act = np.exp(-dist)
        return act


    def _make_link(self, b, s):
        self.G.add_edge(b,s,age = 0)



    def _add_node(self, x, y, b, s):
        r = max(self.G.nodes()) + 1
        pos_r = 0.5 * (x + self.G.nodes[b]['pos'])
        dist = sp.cdist(x, pos_r, metric='euclidean')[0,0]
        act = np.exp(-dist)
        pos_r = pos_r[0,:]
        self.G.add_nodes_from([(r, {'pos' : pos_r, 'fir' : self.sti_s, 'n_best' : 0, 'label' : y,'best_act' : act})])
        self.G.remove_edge(b,s)
        self.G.add_edge(r, b, age = 0)
        self.G.add_edge(r, s, age = 0)
        return r


    def _update_network(self, x, b):
        dpos_b = self.eps_b * self.G.nodes[b]['fir']*(x - self.G.nodes[b]['pos'])
        self.G.nodes[b]['pos'] = self.G.nodes[b]['pos'] + dpos_b[0,:]

        neighbors = self.G.neighbors(b)
        for n in neighbors:

            dpos_n = self.eps_n * self.G.nodes[n]['fir'] * (
                     x - self.G.nodes[n]['pos'])
            self.G.nodes[n]['pos'] = self.G.nodes[n]['pos'] + dpos_n[0,:]



            self.G.edges[b,n]['age'] += 1


    def _update_firing(self, b):

        self.G.nodes[b]['fir'] = self.h_0 - (self.sti_s/self.alpha_b)*\
            (1-np.exp(-self.alpha_b*self.G.nodes[b]['n_best']/self.tau_b))
        






        neighbors = self.G.neighbors(b)
        for n in neighbors:
            self.G.nodes[n]['fir'] = self.h_0 - (self.sti_s/self.alpha_n)*\
                (1-np.exp(-self.alpha_n*self.G.nodes[n]['n_best']/self.tau_n))



    def _remove_old_edges(self):
        for e in self.G.edges():
            if self.G[e[0]][e[1]]['age'] > self.max_age:
                self.G.remove_edge(*e)
        self.G.remove_nodes_from(list(nx.isolates(self.G)))





    def _check_stopping_criterion(self):

        pass

    def _training_step(self, x, y):

        b, s = self._get_best_matching(x)

        self._make_link(b, s)
        act = self._get_activation(x, b)
        fir = self.G.nodes[b]['fir']
        logging.debug('Training step - best matching: %s, %s \n'
                      'Network activation: %s \n'
                      'Firing: %s', str(b), str(s), str(np.round(act,3)),
                      str(np.round(fir,3)))
        if act < self.act_thr and fir < self.fir_thr \
            and len(self.G.nodes()) < self.max_size:
            r = self._add_node(x, y, b, s)
            logging.debug('GENERATE NODE %s', self.G.nodes[r])
        else:
            self._update_network(x, b)
            if ((act > self.G.nodes[b]['best_act']) and (int(self.G.nodes[b]['label']) != int(y))):
                self.G.nodes[b]['label'] = y
                self.num_changes += 1

        self._update_firing(b)
        self._remove_old_edges()


    def train(self, X, Y, n_epochs=20, warm_start = False):
        if not warm_start:
            self._initialize(X,Y)
        for n in range(n_epochs):
            print('gwr epoch: ',n)
            logging.info('>>> Training epoch %s', str(n))
            for i in range(X.shape[0]):
                x = X[i,np.newaxis]
                y = Y[i]
                self._training_step(x,y)
                self._check_stopping_criterion()


        logging.info('Training ended - Network size: %s', len(self.G.nodes())) 
        print('num_changes: ', self.num_changes) 
        self.num_changes = 0    
        return self.G

    def test(self, X, Y,num_tasks):
        num_correct = 0
        class_by_class = np.zeros(num_tasks)
        class_by_class_pred = np.zeros(num_tasks)
        for i in range(X.shape[0]):
            x = X[i,np.newaxis]
            y = Y[i]
            class_by_class[int(y)] += 1
            b,s = self._get_best_matching(x)
            act = self._get_activation(x, b)

            y_pred = self.G.nodes[b]['label']
            if (y_pred == y):
                num_correct += 1
                class_by_class_pred[int(y)] += 1
        return num_correct/len(Y),class_by_class_pred/class_by_class

    def _test_best_matching(self, x):
        pos = self.get_positions()


        dist = sp.cdist(x, pos, metric='euclidean')
        sorted_dist = np.argsort(dist)



        return sorted_dist[0,:10]

    def KNearest_test(self, X, Y):
        num_correct = 0
        class_by_class = np.zeros(10)
        class_by_class_pred = np.zeros(10)
        for i in range(X.shape[0]):
            x = X[i,np.newaxis]
            y = Y[i]
            class_by_class[int(y)] += 1
            best_matches = self._test_best_matching(x)
            votes = np.zeros(10)
            for j in best_matches:
                votes[int(self.G.nodes[j]['label'])] += 1
            y_pred = np.argsort(votes)[-1]
            if (y_pred == y):
                num_correct += 1
                class_by_class_pred[int(y)] += 1
        return num_correct/len(Y),class_by_class_pred/class_by_class

    def choose_task(self, X, num_tasks):
        pred_class = np.zeros(len(X))
        for i in range(X.shape[0]):
            x = X[i,np.newaxis]

            best_matches = self._test_best_matching(x)
            votes = np.zeros(num_tasks)
            for j in best_matches:
                votes[int(self.G.nodes[j]['label'])] += 1
            y_pred = np.argsort(votes)[-1]
            pred_class[i] = y_pred
        return pred_class

    def nodes_per_task(self,num_tasks):
        tasks = np.zeros(num_tasks)
        for node in range(self.G.number_of_nodes()):
            tasks[int(self.G.nodes[node]['label'])] += 1
        return tasks
    


class gwr_torch():

	def __init__(self, act_thr = 0.35, fir_thr = 0.1, eps_b = 0.1,
				 eps_n = 0.01, tau_b = 0.3, tau_n = 0.1, alpha_b = 1.05,
				 alpha_n = 1.05, h_0 = 1, sti_s = 1,
				 max_age = 100, max_size = 100,
				 random_state = None, device = "cpu" ):
		self.act_thr  = act_thr
		self.fir_thr  = fir_thr
		self.eps_b    = eps_b
		self.eps_n    = eps_n
		self.tau_b    = tau_b
		self.tau_n    = tau_n
		self.alpha_b  = alpha_b
		self.alpha_n  = alpha_n
		self.h_0   = h_0
		self.sti_s   = sti_s
		self.max_age  = max_age
		self.max_size = max_size
		self.num_changes = 0
		self.device = device
		if random_state is not None:
			np.random.seed(random_state)

	def _initialize(self, X, Y):
		self.node_id = torch.tensor([]).int()
		self.pos = torch.tensor([]).float()
		self.fir = torch.tensor([]).float()
		self.n_best = torch.tensor([]).int()
		self.label = torch.tensor([]).float()
		self.best_act = torch.tensor([]).float()
		self.edges = {}

		draw = torch.randint(X.shape[0],(2,))


		self.node_id = torch.cat((self.node_id,torch.tensor([0]).int()))
		self.pos = torch.cat((self.pos,X[draw[0],:].view(1,-1)))
		self.label = torch.cat((self.label,Y[draw[0]].view(1)))
		self.fir = torch.cat((self.fir,torch.tensor([self.sti_s]).float()))
		self.n_best = torch.cat((self.n_best,torch.tensor([0]).int()))
		self.best_act = torch.cat((self.best_act,torch.tensor([1]).float()))
		self.edges[0] = {} 


		self.node_id = torch.cat((self.node_id,torch.tensor([1]).int()))
		self.pos = torch.cat((self.pos,X[draw[1],:].view(1,-1)))
		self.label = torch.cat((self.label,Y[draw[1]].view(1)))
		self.fir = torch.cat((self.fir,torch.tensor([self.sti_s]).float()))
		self.n_best = torch.cat((self.n_best,torch.tensor([0]).int()))
		self.best_act = torch.cat((self.best_act,torch.tensor([1]).float()))
		self.edges[1] = {} 

		self.last_node_idx = 1




	def _get_best_matching(self, x):  
		dist = torch.cdist(x.view(1,-1), self.pos)
		sorted_dist = torch.argsort(dist.view(-1))

		b = self.node_id[sorted_dist[0]]
		s = self.node_id[sorted_dist[1]]
		id_b = sorted_dist[0]
		id_s = sorted_dist[1]

		self.n_best[sorted_dist[0]] += 1

		return id_b, id_s


	def _get_activation(self, x, id_b):
		pos_b = self.pos[id_b]
		dist = torch.dist(x, pos_b)

		act = torch.exp(-dist)
		return act


	def _make_link(self, id_b, id_s):
		b = int(self.node_id[id_b])
		s = int(self.node_id[id_s])
		if(s not in self.edges[b]):
			self.edges[b][s] = 0
			self.edges[s][b] = 0 

	def _add_node(self, x, y, id_b, id_s):
		r = self.last_node_idx + 1
		pos_r = 0.5 * (x + self.pos[id_b])
		dist = torch.dist(x, pos_r)
		act = torch.exp(-dist)


		self.node_id = torch.cat((self.node_id,torch.tensor([r]).int()))
		self.pos = torch.cat((self.pos,pos_r.view(1,-1)))
		self.label = torch.cat((self.label,y.view(1)))
		self.fir = torch.cat((self.fir,torch.tensor([self.sti_s]).float()))
		self.n_best = torch.cat((self.n_best,torch.tensor([0]).int()))
		self.best_act = torch.cat((self.best_act,torch.tensor([act]).float()))
		self.edges[r] = {}


		b = int(self.node_id[id_b])
		s = int(self.node_id[id_s])
		if(s in self.edges[b]):
			del self.edges[b][s]
			del self.edges[s][b]


		self.edges[r][b] = 0
		self.edges[r][s] = 0
		self.edges[b][r] = 0
		self.edges[s][r] = 0

		self.last_node_idx = r
		return r

	def _update_network(self, x, id_b):
		dpos_b = self.eps_b * self.fir[id_b]*(x - self.pos[id_b])
		self.pos[id_b] = self.pos[id_b] + dpos_b

		b = int(self.node_id[id_b])
		for n in self.edges[b]:
			id_n = (self.node_id==n).nonzero(as_tuple = True)[0][0]

			dpos_n = self.eps_n * self.fir[id_n]*(x - self.pos[id_n])
			self.pos[id_n] = self.pos[id_n] + dpos_n  

			self.edges[b][n] += 1                   
			self.edges[n][b] += 1 

	def _update_firing(self, id_b):


		dfir_b = (1/self.tau_b) * self.alpha_b*(1-self.fir[id_b]) - (1/self.tau_b)   
		self.fir[id_b] = torch.clamp(self.fir[id_b] + dfir_b,0.001,1)                   
		b = int(self.node_id[id_b])
		for n in list(self.edges[b]):
			id_n = (self.node_id==n).nonzero(as_tuple = True)[0][0]
			dfir_n = (1/self.tau_n) * self.alpha_n*(1-self.fir[id_n]) - (1/self.tau_n)
			self.fir[id_n] = torch.clamp(self.fir[id_n] + dfir_n,0.001,1)          



	def _remove_old_edges(self):
		for node_i in list(self.edges):
			for node_j in list(self.edges[node_i]):
				if(self.edges[node_i][node_j] > self.max_age):
					del self.edges[node_i][node_j]
					del self.edges[node_j][node_i]
		for node_i in list(self.edges):
			if(len(self.edges[node_i]) == 0):
				id_node_i = (self.node_id==node_i).nonzero(as_tuple = True)[0][0]
				self.node_id = torch.cat((self.node_id[:id_node_i],self.node_id[id_node_i+1:]))
				self.pos = torch.cat((self.pos[:id_node_i],self.pos[id_node_i+1:]))
				self.label = torch.cat((self.label[:id_node_i],self.label[id_node_i+1:]))
				self.fir = torch.cat((self.fir[:id_node_i],self.fir[id_node_i+1:]))
				self.n_best = torch.cat((self.n_best[:id_node_i],self.n_best[id_node_i+1:]))
				self.best_act = torch.cat((self.best_act[:id_node_i],self.best_act[id_node_i+1:]))
				del self.edges[node_i]

	def _check_stopping_criterion(self):

		pass

	def _training_step(self, x, y):
		id_b, id_s = self._get_best_matching(x)

		self._make_link(id_b, id_s)
		act = self._get_activation(x, id_b)
		fir = self.fir[id_b]

		if act < self.act_thr and fir < self.fir_thr \
			and len(self.node_id) < self.max_size:
			r = self._add_node(x, y, id_b, id_s)

		else:
			self._update_network(x, id_b)
			act = self._get_activation(x, id_b)
			if ((act > self.best_act[id_b]) and (int(self.label[id_b]) != int(y))):
				self.label[id_b] = y
				self.num_changes += 1

		self._update_firing(id_b)
		self._remove_old_edges()


	def train(self, X, Y, n_epochs=20, warm_start = False):
		if not warm_start:
			self._initialize(X,Y)
		print('training GWR..')

		for n in range(n_epochs):


			for i in range(X.shape[0]):
				x = X[i]
				y = Y[i]
				self._training_step(x,y)
				self._check_stopping_criterion()
		print('num_changes: ', self.num_changes)
		self.num_changes = 0    

	def test(self, X, Y,num_tasks):                   
		num_correct = 0
		class_by_class = np.zeros(num_tasks)
		class_by_class_pred = np.zeros(num_tasks)
		for i in range(X.shape[0]):
			x = X[i]
			y = Y[i]
			class_by_class[int(y)] += 1
			id_b,id_s = self._get_best_matching(x)
			act = self._get_activation(x,id_b)
			y_pred = self.label[id_b]
			if (y_pred == y):
				num_correct += 1
				class_by_class_pred[int(y)] += 1
		return num_correct/len(Y),class_by_class_pred/class_by_class
	  
	def _test_best_matching(self, x,k):          
		dist = torch.cdist(x.view(1,-1), self.pos)
		sorted_dist = torch.argsort(dist.view(-1))
		return sorted_dist[:k]
 
	def KNearest_test(self, X, Y,num_tasks,k):                   
		num_correct = 0
		class_by_class = np.zeros(num_tasks)
		class_by_class_pred = np.zeros(num_tasks)
		for i in range(X.shape[0]):
			x = X[i]
			y = Y[i]
			class_by_class[int(y)] += 1
			best_matches = self._test_best_matching(x,k)
			votes = np.zeros(num_tasks)
			for j in best_matches:
				votes[int(self.label[j])] += 1
			y_pred = np.argsort(votes)[-1]
			if (y_pred == y):
				num_correct += 1
				class_by_class_pred[int(y)] += 1
		return num_correct/len(Y),class_by_class_pred/class_by_class

	def choose_task(self, X, num_tasks,k):
		pred_class = np.zeros(len(X))
		for i in range(X.shape[0]):
			x = X[i]

			best_matches = self._test_best_matching(x,k)
			votes = np.zeros(num_tasks)
			for j in best_matches:
				votes[int(self.label[j])] += 1
			y_pred = np.argsort(votes)[-1]

			pred_class[i] = y_pred
		return pred_class
		
	def nodes_per_task(self,num_tasks):
		tasks = np.zeros(num_tasks)
		for l in self.label:
			tasks[int(l)] += 1
		return tasks

	def get_num_nodes(self):
		return len(self.node_id)




class gwr_task_torch():

	def __init__(self, act_thr = 0.35, fir_thr = 0.1, eps_b = 0.1,
				 eps_n = 0.01, tau_b = 0.3, tau_n = 0.1, alpha_b = 1.05,
				 alpha_n = 1.05, h_0 = 1, sti_s = 1,
				 max_age = 100, max_size = 100,
				 random_state = None, device = "cpu" ):
		self.act_thr  = act_thr
		self.fir_thr  = fir_thr
		self.eps_b    = eps_b
		self.eps_n    = eps_n
		self.tau_b    = tau_b
		self.tau_n    = tau_n
		self.alpha_b  = alpha_b
		self.alpha_n  = alpha_n
		self.h_0   = h_0
		self.sti_s   = sti_s
		self.max_age  = max_age
		self.max_size = max_size
		self.num_changes = 0
		self.device = device
		if random_state is not None:
			np.random.seed(random_state)

		self.node_id = {}
		self.pos = {}
		self.fir = {}
		self.n_best = {}
		self.label = {}
		self.best_act = {}
		self.edges = {}
		self.last_node_idx = {}


	def _initialize(self, X, Y, task_id):
		self.node_id[task_id] = torch.tensor([]).int()
		self.pos[task_id] = torch.tensor([]).float()
		self.fir[task_id] = torch.tensor([]).float()
		self.n_best[task_id] = torch.tensor([]).int()
		self.label[task_id] = torch.tensor([]).float()
		self.best_act[task_id] = torch.tensor([]).float()
		self.edges[task_id] = {}
		self.last_node_idx[task_id] = 0
		draw = torch.randint(X.shape[0],(2,))



		self.node_id[task_id] = torch.cat((self.node_id[task_id],torch.tensor([self.last_node_idx[task_id]]).int()))
		self.pos[task_id] = torch.cat((self.pos[task_id],X[draw[0],:].view(1,-1)))
		self.label[task_id] = torch.cat((self.label[task_id],Y[draw[0]].view(1)))
		self.fir[task_id] = torch.cat((self.fir[task_id],torch.tensor([self.sti_s]).float()))
		self.n_best[task_id] = torch.cat((self.n_best[task_id],torch.tensor([0]).int()))
		self.best_act[task_id] = torch.cat((self.best_act[task_id],torch.tensor([1]).float()))
		self.edges[task_id][0] = {} 
		self.last_node_idx[task_id] += 1


		self.node_id[task_id] = torch.cat((self.node_id[task_id],torch.tensor([self.last_node_idx[task_id]]).int()))
		self.pos[task_id] = torch.cat((self.pos[task_id],X[draw[1],:].view(1,-1)))
		self.label[task_id] = torch.cat((self.label[task_id],Y[draw[1]].view(1)))
		self.fir[task_id] = torch.cat((self.fir[task_id],torch.tensor([self.sti_s]).float()))
		self.n_best[task_id] = torch.cat((self.n_best[task_id],torch.tensor([0]).int()))
		self.best_act[task_id] = torch.cat((self.best_act[task_id],torch.tensor([1]).float()))
		self.edges[task_id][1] = {} 






	def _get_best_matching(self, x,task_id):  
		dist = torch.cdist(x.view(1,-1), self.pos[task_id],p=1)
		sorted_dist = torch.argsort(dist.view(-1))

		b = self.node_id[task_id][sorted_dist[0]]
		s = self.node_id[task_id][sorted_dist[1]]
		id_b = sorted_dist[0]
		id_s = sorted_dist[1]

		self.n_best[task_id][sorted_dist[0]] += 1

		return id_b, id_s


	def _get_activation(self, x, id_b,task_id):
		pos_b = self.pos[task_id][id_b]
		dist = torch.dist(x, pos_b,p=1)

		act = torch.exp(-dist)
		return act


	def _make_link(self, id_b, id_s,task_id):
		b = int(self.node_id[task_id][id_b])
		s = int(self.node_id[task_id][id_s])
		if(s not in self.edges[task_id][b]):
			self.edges[task_id][b][s] = 0
			self.edges[task_id][s][b] = 0 

	def _add_node(self, x, y, id_b, id_s,task_id):
		r = self.last_node_idx[task_id] + 1
		pos_r = 0.5 * (x + self.pos[task_id][id_b])
		dist = torch.dist(x, pos_r)
		act = torch.exp(-dist)


		self.node_id[task_id] = torch.cat((self.node_id[task_id],torch.tensor([r]).int()))
		self.pos[task_id] = torch.cat((self.pos[task_id],pos_r.view(1,-1)))
		self.label[task_id] = torch.cat((self.label[task_id],y.view(1)))
		self.fir[task_id] = torch.cat((self.fir[task_id],torch.tensor([self.sti_s]).float()))
		self.n_best[task_id] = torch.cat((self.n_best[task_id],torch.tensor([0]).int()))
		self.best_act[task_id] = torch.cat((self.best_act[task_id],torch.tensor([act]).float()))
		self.edges[task_id][r] = {}


		b = int(self.node_id[task_id][id_b])
		s = int(self.node_id[task_id][id_s])
		if(s in self.edges[task_id][b]):
			del self.edges[task_id][b][s]
			del self.edges[task_id][s][b]


		self.edges[task_id][r][b] = 0
		self.edges[task_id][r][s] = 0
		self.edges[task_id][b][r] = 0
		self.edges[task_id][s][r] = 0

		self.last_node_idx[task_id] = r
		return r

	def _update_network(self, x, id_b,task_id):
		dpos_b = self.eps_b * self.fir[task_id][id_b]*(x - self.pos[task_id][id_b])
		self.pos[task_id][id_b] = self.pos[task_id][id_b] + dpos_b

		b = int(self.node_id[task_id][id_b])
		for n in self.edges[task_id][b]:
			id_n = (self.node_id[task_id]==n).nonzero(as_tuple = True)[0][0]

			dpos_n = self.eps_n * self.fir[task_id][id_n]*(x - self.pos[task_id][id_n])
			self.pos[task_id][id_n] = self.pos[task_id][id_n] + dpos_n  

			self.edges[task_id][b][n] += 1                   
			self.edges[task_id][n][b] += 1 

	def _update_firing(self, id_b,task_id):


		dfir_b = (1/self.tau_b) * self.alpha_b*(1-self.fir[task_id][id_b]) - (1/self.tau_b)   
		self.fir[task_id][id_b] = torch.clamp(self.fir[task_id][id_b] + dfir_b,0.001,1)                   
		b = int(self.node_id[task_id][id_b])
		for n in list(self.edges[task_id][b]):
			id_n = (self.node_id[task_id]==n).nonzero(as_tuple = True)[0][0]
			dfir_n = (1/self.tau_n) * self.alpha_n*(1-self.fir[task_id][id_n]) - (1/self.tau_n)
			self.fir[task_id][id_n] = torch.clamp(self.fir[task_id][id_n] + dfir_n,0.001,1)          



	def _remove_old_edges(self):
		for i,edges in self.edges.items():
			for node_i in list(edges):
				for node_j in list(edges[node_i]):
					if(edges[node_i][node_j] > self.max_age):
						del edges[node_i][node_j]
						del edges[node_j][node_i]
			for node_i in list(edges):
				if(len(edges[node_i]) == 0):
					id_node_i = (self.node_id[i]==node_i).nonzero(as_tuple = True)[0][0]
					self.node_id[i] = torch.cat((self.node_id[i][:id_node_i],self.node_id[i][id_node_i+1:]))
					self.pos[i] = torch.cat((self.pos[i][:id_node_i],self.pos[i][id_node_i+1:]))
					self.label[i] = torch.cat((self.label[i][:id_node_i],self.label[i][id_node_i+1:]))
					self.fir[i] = torch.cat((self.fir[i][:id_node_i],self.fir[i][id_node_i+1:]))
					self.n_best[i] = torch.cat((self.n_best[i][:id_node_i],self.n_best[i][id_node_i+1:]))
					self.best_act[i] = torch.cat((self.best_act[i][:id_node_i],self.best_act[i][id_node_i+1:]))
					del edges[node_i]

	def _check_stopping_criterion(self):

		pass

	def _training_step(self, x, y,task_id):
		id_b, id_s = self._get_best_matching(x,task_id)

		self._make_link(id_b, id_s,task_id)
		act = self._get_activation(x, id_b,task_id)
		fir = self.fir[task_id][id_b]

		if act < self.act_thr and fir < self.fir_thr \
			and len(self.node_id[task_id]) < self.max_size:
			r = self._add_node(x, y, id_b, id_s,task_id)

		else:
			self._update_network(x, id_b,task_id)





		self._update_firing(id_b,task_id)
		self._remove_old_edges()


	def train(self, X, Y, task_id, n_epochs=20, new_task = False):
		if new_task:
			self._initialize(X,Y,task_id)
		print('training GWR..')
		act_thr = self.act_thr
		self.act_thr = np.exp(-4)

		feature_count = 0

		for n in range(n_epochs):


			for i in range(X.shape[0]):
				x = X[i]
				y = Y[i]
				self._training_step(x,y,task_id)
				self._check_stopping_criterion()
				feature_count += 1
				if(feature_count>50):
					self.act_thr = act_thr



	def test(self, X, Y,num_tasks):                   
		num_correct = 0
		class_by_class = np.zeros(num_tasks)
		class_by_class_pred = np.zeros(num_tasks)
		for i in range(X.shape[0]):
			x = X[i]
			y = Y[i]
			class_by_class[int(y)] += 1
			id_b,id_s = self._get_best_matching(x)
			act = self._get_activation(x,id_b)
			y_pred = self.label[id_b]
			if (y_pred == y):
				num_correct += 1
				class_by_class_pred[int(y)] += 1
		return num_correct/len(Y),class_by_class_pred/class_by_class
	  
	def _test_best_matching(self, x,k):
		Dist = torch.tensor([])
		for i,pos in self.pos.items():


			dist = torch.cdist(x.view(1,-1), pos, p=1)
			sorted_dist,sort_id = torch.sort(dist.view(-1))
			Dist = torch.cat((Dist,sorted_dist[:k]))
		sorted_Dist,sorted_ids = torch.sort(Dist.view(-1))

		channel_wise_thres = torch.zeros(k)
		for i,ids in enumerate(sorted_ids[:k]):
			channel_wise_thres[i] += torch.sum(torch.where(torch.exp(-torch.nan_to_num(torch.sqrt(x**2-pos[ids//k][ids%k]**2), nan=1e5)) > 0.5,1,0))

		sorted_ids = sorted_ids // k
		return sorted_ids[:k], channel_wise_thres
 
	def KNearest_test(self, X, Y,num_tasks,k):                   
		num_correct = 0
		class_by_class = np.zeros(num_tasks)
		class_by_class_pred = np.zeros(num_tasks)
		for i in range(X.shape[0]):
			x = X[i]
			y = Y[i]
			class_by_class[int(y)] += 1
			best_matches = self._test_best_matching(x,k)

			votes = np.zeros(num_tasks)
			for j in best_matches:
				votes[j] += 1
			y_pred = np.argsort(votes)[-1]
			if (y_pred == y):
				num_correct += 1
				class_by_class_pred[int(y)] += 1
		return num_correct/len(Y),class_by_class_pred/class_by_class






















	def choose_task(self, X, num_tasks,k):


		for i in range(X.shape[0]):
			x = X[i]

			best_matches, distances = self._test_best_matching(x,k)
			position_importance = torch.tensor([(1.1-i/10) for i in range(1,k+1)])
			votes = torch.zeros(num_tasks, dtype=torch.float32)
			dist_sum = torch.zeros(num_tasks, dtype=torch.float32)



			task_occ = torch.argsort(distances)
			distances = torch.nn.functional.softmax(distances,dim=0)
			y_pred = {int(best_matches[task_occ[-1]]):distances[int(task_occ[-1])],int(best_matches[task_occ[-2]]):distances[int(task_occ[-2])]}







		return y_pred
		
	def nodes_per_task(self,num_tasks):
		tasks = np.zeros(num_tasks)
		for n,n_id in self.node_id.items():
			tasks[n] = len(n_id)			
		return tasks

	def get_num_nodes(self):
		return len(self.node_id)

	def getpos(self):
		return self.pos

	def getlabels(self):
		return self.label

