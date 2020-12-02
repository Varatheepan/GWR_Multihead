from __future__ import division

import logging

import copy
import numpy as np
import networkx as nx
import scipy.spatial.distance as sp
import matplotlib.pyplot as plt

# import torch
# import time

# TODO should have a _get_activation_trajectories method too

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
        # TODO: initialize empty labels?
        draw = np.random.choice(X.shape[0], size=2, replace=False)
        # print(X[draw[0]].shape)                                   #vara
        # print(draw)                                               #vara
        self.G.add_nodes_from([(0,{'pos' : X[draw[0],:],'fir' : 1, 'label' : Y[draw[0]]})])      #vara
        self.G.add_nodes_from([(1,{'pos' : X[draw[1],:],'fir' : 1, 'label' : Y[draw[1]]})])      #vara
        #print(self.G.number_of_nodes())                            #vara
        #print(nx.get_node_attributes(self.G, 'pos').values())      #vara
        #print(self.G.nodes[0])                                     #vara

    def get_positions(self):
        pos = np.array(list(nx.get_node_attributes(self.G, 'pos').values()))    #vara
        return pos

    def _get_best_matching(self, x):
        pos = self.get_positions()
        #np.concatenate(pos,axis = 0)               #vara
        #print(pos)           
        # input('dist1...')  
        # t1 = time.time()                    #vara
        # dist = sp.cdist(x, pos, metric='euclidean')
        # print('time..',time.time()-t1)
        # input('dist2...')
        # t2 = time.time()   
        # x1 = torch.from_numpy(x)
        # pos1 = torch.from_numpy(pos)
        # dist1 = torch.sqrt(torch.pow(x1-pos1,2).sum(1))
        # dist1 = dist1.numpy()
        # print('time..',time.time()-t2)
        # input(...)
        dist = sp.cdist(x, pos, metric='euclidean')
        sorted_dist = np.argsort(dist)
        #print('sroted dist' , sorted_dist)         #vara
        b = sorted_dist[0,0]
        s = sorted_dist[0,1]
        return b, s


    def _get_activation(self, x, b):
        p = self.G.nodes[b]['pos'][np.newaxis,:]
        dist = sp.cdist(x, p, metric='euclidean')[0,0]
        act = np.exp(-dist)
        return act


    def _make_link(self, b, s):
        self.G.add_edge(b,s,age = 0)            #vara
        #print('edges ',self.G.edges([0]))      #vara


    def _add_node(self, x, y, b, s):
        r = max(self.G.nodes()) + 1
        pos_r = 0.5 * (x + self.G.nodes[b]['pos'])[0,:]
        self.G.add_nodes_from([(r, {'pos' : pos_r, 'fir' : 1, 'label' : y})])    #vara
        self.G.remove_edge(b,s)
        self.G.add_edge(r, b, age = 0)          #vara
        self.G.add_edge(r, s, age = 0)          #vara
        return r


    def _update_network(self, x, b):
        dpos_b = self.eps_b * self.G.nodes[b]['fir']*(x - self.G.nodes[b]['pos'])
        self.G.nodes[b]['pos'] = self.G.nodes[b]['pos'] + dpos_b[0,:]

        neighbors = self.G.neighbors(b)
        for n in neighbors:
            # update the position of the neighbors
            dpos_n = self.eps_n * self.G.nodes[n]['fir'] * (
                     x - self.G.nodes[n]['pos'])
            self.G.nodes[n]['pos'] = self.G.nodes[n]['pos'] + dpos_n[0,:]

            # increase the age of all edges connected to b
            #print('age ', self.G.edges[b,n]['age'])        #vara
            self.G.edges[b,n]['age'] += 1                   #vara


    def _update_firing(self, b):
        dfir_b = self.tau_b * self.kappa*(1-self.G.nodes[b]['fir']) - self.tau_b        #different
        self.G.nodes[b]['fir'] = self.G.nodes[b]['fir']  + dfir_b
        # if(self.G.nodes[b]['fir']<0):
        #     print("negtive problem")
        self.G.nodes[b]['fir'] = np.clip(self.G.nodes[b]['fir'],0,1) #max(0,self.G.nodes[b]['fir']);
        #print('fir ',self.G.nodes[b]['fir'])   #vara
        #print(self.G.nodes)                    #vara

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
        # TODO: implement this
        pass

    def _training_step(self, x, y):
        # TODO: do not recompute all positions at every iteration
        b, s = self._get_best_matching(x)
        #print('best match ',b,s)            #vara
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
        #nx.draw(self.G)        #vara
        #plt.show()
        logging.info('Training ended - Network size: %s', len(self.G.nodes()))      
        return self.G

    def test(self, X, Y):                   #vara
        num_correct = 0
        class_by_class = np.zeros(10)
        class_by_class_pred = np.zeros(10)
        for i in range(X.shape[0]):
            x = X[i,np.newaxis]
            y = Y[i]
            # print('y: ',int(y))
            class_by_class[int(y)] += 1
            b,s = self._get_best_matching(x)
            act = self._get_activation(x, b)
            # print('activation : ', act)
            y_pred = self.G.nodes[b]['label']
            if (y_pred == y):
                num_correct += 1
                class_by_class_pred[int(y)] += 1
        return num_correct/len(Y),class_by_class_pred/class_by_class

    def _test_best_matching(self, x):
        pos = self.get_positions()
        #np.concatenate(pos,axis = 0)               #vara
        #print(pos)           
        dist = sp.cdist(x, pos, metric='euclidean')
        sorted_dist = np.argsort(dist)
        #print('sroted dist' , sorted_dist)         #vara
        # b = sorted_dist[0,0]
        # s = sorted_dist[0,1]
        return sorted_dist[0,:10]

    def KNearest_test(self, X, Y):                   #vara
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
            # print('votes : ', votes)
            # print(y_pred)
            # input('....')
            if (y_pred == y):
                num_correct += 1
                class_by_class_pred[int(y)] += 1
        return num_correct/len(Y),class_by_class_pred/class_by_class

    def choose_task(self, X, num_tasks):
        pred_class = np.zeros(len(X))
        for i in range(X.shape[0]):
            x = X[i,np.newaxis]
            # y = Y[i]
            best_matches = self._test_best_matching(x)
            votes = np.zeros(num_tasks)
            for j in best_matches:
                votes[int(self.G.nodes[j]['label'])] += 1
            y_pred = np.argsort(votes)[-1]
            pred_class[i] = y_pred
        return pred_class

## changing label according to last known best activation
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
        # TODO: initialize empty labels?
        draw = np.random.choice(X.shape[0], size=2, replace=False)
        # print(X[draw[0]].shape)                                   #vara
        # print(draw)                                               #vara
        self.G.add_nodes_from([(0,{'pos' : X[draw[0],:],'fir' : 1, 'label' : Y[draw[0]], 'best_act' : 1})])      #vara
        self.G.add_nodes_from([(1,{'pos' : X[draw[1],:],'fir' : 1, 'label' : Y[draw[1]], 'best_act' : 1})])      #vara
        #print(self.G.number_of_nodes())                            #vara
        #print(nx.get_node_attributes(self.G, 'pos').values())      #vara
        #print(self.G.nodes[0])                                     #vara

    def get_positions(self):
        pos = np.array(list(nx.get_node_attributes(self.G, 'pos').values()))    #vara
        return pos

    def _get_best_matching(self, x):
        pos = self.get_positions()
        #np.concatenate(pos,axis = 0)               #vara
        #print(pos)           
        dist = sp.cdist(x, pos, metric='euclidean')
        sorted_dist = np.argsort(dist)
        #print('sroted dist' , sorted_dist)         #vara
        b = sorted_dist[0,0]
        s = sorted_dist[0,1]
        return b, s


    def _get_activation(self, x, b):
        p = self.G.nodes[b]['pos'][np.newaxis,:]
        dist = sp.cdist(x, p, metric='euclidean')[0,0]
        act = np.exp(-dist)
        return act


    def _make_link(self, b, s):
        self.G.add_edge(b,s,age = 0)            #vara
        #print('edges ',self.G.edges([0]))      #vara


    def _add_node(self, x, y, b, s):
        r = max(self.G.nodes()) + 1
        pos_r = 0.5 * (x + self.G.nodes[b]['pos'])
        dist = sp.cdist(x, pos_r, metric='euclidean')[0,0]
        act = np.exp(-dist)
        pos_r = pos_r[0,:]
        self.G.add_nodes_from([(r, {'pos' : pos_r, 'fir' : 1, 'label' : y,'best_act' : act})])    #vara
        self.G.remove_edge(b,s)
        self.G.add_edge(r, b, age = 0)          #vara
        self.G.add_edge(r, s, age = 0)          #vara
        return r


    def _update_network(self, x, b):
        dpos_b = self.eps_b * self.G.nodes[b]['fir']*(x - self.G.nodes[b]['pos'])
        self.G.nodes[b]['pos'] = self.G.nodes[b]['pos'] + dpos_b[0,:]

        neighbors = self.G.neighbors(b)
        for n in neighbors:
            # update the position of the neighbors
            dpos_n = self.eps_n * self.G.nodes[n]['fir'] * (
                     x - self.G.nodes[n]['pos'])
            self.G.nodes[n]['pos'] = self.G.nodes[n]['pos'] + dpos_n[0,:]

            # increase the age of all edges connected to b
            #print('age ', self.G.edges[b,n]['age'])        #vara
            self.G.edges[b,n]['age'] += 1                   #vara


    def _update_firing(self, b):
        dfir_b = self.tau_b * self.kappa*(1-self.G.nodes[b]['fir']) - self.tau_b        #different
        self.G.nodes[b]['fir'] = self.G.nodes[b]['fir']  + dfir_b
        # if(self.G.nodes[b]['fir']<0):
        #     print("negtive problem")
        self.G.nodes[b]['fir'] = np.clip(self.G.nodes[b]['fir'],0,1) #max(0,self.G.nodes[b]['fir']);
        #print('fir ',self.G.nodes[b]['fir'])   #vara
        #print(self.G.nodes)                    #vara

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
        # for node in self.G.nodes():
        #     if len(self.G.edges(node)) == 0:
        #         logging.debug('Removing node %s', str(node))
        #         self.G.remove_node(node)

    def _check_stopping_criterion(self):
        # TODO: implement this
        pass

    def _training_step(self, x, y):
        # TODO: do not recompute all positions at every iteration
        b, s = self._get_best_matching(x)
        #print('best match ',b,s)            #vara
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
        #nx.draw(self.G)        #vara
        #plt.show()
        logging.info('Training ended - Network size: %s', len(self.G.nodes())) 
        print('num_changes: ', self.num_changes) 
        self.num_changes = 0    
        return self.G

    def test(self, X, Y):                   #vara
        num_correct = 0
        class_by_class = np.zeros(10)
        class_by_class_pred = np.zeros(10)
        for i in range(X.shape[0]):
            x = X[i,np.newaxis]
            y = Y[i]
            class_by_class[int(y)] += 1
            b,s = self._get_best_matching(x)
            act = self._get_activation(x, b)
            # print('activation : ', act)
            y_pred = self.G.nodes[b]['label']
            if (y_pred == y):
                num_correct += 1
                class_by_class_pred[int(y)] += 1
        return num_correct/len(Y),class_by_class_pred/class_by_class

    def _test_best_matching(self, x):
        pos = self.get_positions()
        #np.concatenate(pos,axis = 0)               #vara
        #print(pos)           
        dist = sp.cdist(x, pos, metric='euclidean')
        sorted_dist = np.argsort(dist)
        #print('sroted dist' , sorted_dist)         #vara
        # b = sorted_dist[0,0]
        # s = sorted_dist[0,1]
        return sorted_dist[0,:10]

    def KNearest_test(self, X, Y):                   #vara
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
            # y = Y[i]
            best_matches = self._test_best_matching(x)
            votes = np.zeros(num_tasks)
            for j in best_matches:
                votes[int(self.G.nodes[j]['label'])] += 1
            y_pred = np.argsort(votes)[-1]
            pred_class[i] = y_pred
        return pred_class


## Firing update changed as original
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
        # TODO: initialize empty labels?
        draw = np.random.choice(X.shape[0], size=2, replace=False)
        # print(X[draw[0]].shape)                                   #vara
        # print(draw)                                               #vara
        self.G.add_nodes_from([(0,{'pos' : X[draw[0],:],'fir' : self.sti_s, 'n_best' : 0, 'label' : Y[draw[0]], 'best_act' : 1})])      #vara
        self.G.add_nodes_from([(1,{'pos' : X[draw[1],:],'fir' : self.sti_s, 'n_best' : 0, 'label' : Y[draw[1]], 'best_act' : 1})])      #vara
        #print(self.G.number_of_nodes())                            #vara
        #print(nx.get_node_attributes(self.G, 'pos').values())      #vara
        #print(self.G.nodes[0])                                     #vara

    def get_positions(self):
        pos = np.array(list(nx.get_node_attributes(self.G, 'pos').values()))    #vara
        return pos

    def _get_best_matching(self, x):
        pos = self.get_positions()
        #np.concatenate(pos,axis = 0)               #vara
        #print(pos)           
        dist = sp.cdist(x, pos, metric='euclidean')
        sorted_dist = np.argsort(dist)
        #print('sroted dist' , sorted_dist)         #vara
        b = sorted_dist[0,0]
        s = sorted_dist[0,1]

        self.G.nodes[b]['n_best'] += 1

        return b, s


    def _get_activation(self, x, b):
        p = self.G.nodes[b]['pos'][np.newaxis,:]
        dist = sp.cdist(x, p, metric='euclidean')[0,0]
        # print('x ',x)
        # print('p ',p)
        # print('dist ',dist)
        act = np.exp(-dist)
        return act


    def _make_link(self, b, s):
        self.G.add_edge(b,s,age = 0)            #vara
        #print('edges ',self.G.edges([0]))      #vara


    def _add_node(self, x, y, b, s):
        r = max(self.G.nodes()) + 1
        pos_r = 0.5 * (x + self.G.nodes[b]['pos'])
        dist = sp.cdist(x, pos_r, metric='euclidean')[0,0]
        act = np.exp(-dist)
        pos_r = pos_r[0,:]
        self.G.add_nodes_from([(r, {'pos' : pos_r, 'fir' : self.sti_s, 'n_best' : 0, 'label' : y,'best_act' : act})])    #vara
        self.G.remove_edge(b,s)
        self.G.add_edge(r, b, age = 0)          #vara
        self.G.add_edge(r, s, age = 0)          #vara
        return r


    def _update_network(self, x, b):
        dpos_b = self.eps_b * self.G.nodes[b]['fir']*(x - self.G.nodes[b]['pos'])
        self.G.nodes[b]['pos'] = self.G.nodes[b]['pos'] + dpos_b[0,:]

        neighbors = self.G.neighbors(b)
        for n in neighbors:
            # update the position of the neighbors
            dpos_n = self.eps_n * self.G.nodes[n]['fir'] * (
                     x - self.G.nodes[n]['pos'])
            self.G.nodes[n]['pos'] = self.G.nodes[n]['pos'] + dpos_n[0,:]

            # increase the age of all edges connected to b
            #print('age ', self.G.edges[b,n]['age'])        #vara
            self.G.edges[b,n]['age'] += 1                   #vara


    def _update_firing(self, b):
        # dfir_b = self.tau_b * self.kappa*(1-self.G.nodes[b]['fir']) - self.tau_b        #different
        self.G.nodes[b]['fir'] = self.h_0 - (self.sti_s/self.alpha_b)*\
            (1-np.exp(-self.alpha_b*self.G.nodes[b]['n_best']/self.tau_b))          #vara
        
        # if(self.G.nodes[b]['fir']<0):         #vara
        #     print("negtive problem")
        # self.G.nodes[b]['fir'] = np.clip(self.G.nodes[b]['fir'],0,1) #max(0,self.G.nodes[b]['fir']);
        #print('fir ',self.G.nodes[b]['fir'])   #vara
        #print(self.G.nodes)                    #vara

        neighbors = self.G.neighbors(b)
        for n in neighbors:
            self.G.nodes[n]['fir'] = self.h_0 - (self.sti_s/self.alpha_n)*\
                (1-np.exp(-self.alpha_n*self.G.nodes[n]['n_best']/self.tau_n))
            # self.G.nodes[n]['fir'] = np.clip(self.G.nodes[n]['fir'],0,1)


    def _remove_old_edges(self):
        for e in self.G.edges():
            if self.G[e[0]][e[1]]['age'] > self.max_age:
                self.G.remove_edge(*e)
        self.G.remove_nodes_from(list(nx.isolates(self.G)))
        # for node in self.G.nodes():
        #     if len(self.G.edges(node)) == 0:
        #         logging.debug('Removing node %s', str(node))
        #         self.G.remove_node(node)

    def _check_stopping_criterion(self):
        # TODO: implement this
        pass

    def _training_step(self, x, y):
        # TODO: do not recompute all positions at every iteration
        b, s = self._get_best_matching(x)
        #print('best match ',b,s)            #vara
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
        #nx.draw(self.G)        #vara
        #plt.show()
        logging.info('Training ended - Network size: %s', len(self.G.nodes())) 
        print('num_changes: ', self.num_changes) 
        self.num_changes = 0    
        return self.G

    def test(self, X, Y,num_tasks):                   #vara
        num_correct = 0
        class_by_class = np.zeros(num_tasks)
        class_by_class_pred = np.zeros(num_tasks)
        for i in range(X.shape[0]):
            x = X[i,np.newaxis]
            y = Y[i]
            class_by_class[int(y)] += 1
            b,s = self._get_best_matching(x)
            act = self._get_activation(x, b)
            # print('activation : ', act)
            y_pred = self.G.nodes[b]['label']
            if (y_pred == y):
                num_correct += 1
                class_by_class_pred[int(y)] += 1
        return num_correct/len(Y),class_by_class_pred/class_by_class

    def _test_best_matching(self, x):
        pos = self.get_positions()
        #np.concatenate(pos,axis = 0)               #vara
        #print(pos)           
        dist = sp.cdist(x, pos, metric='euclidean')
        sorted_dist = np.argsort(dist)
        #print('sroted dist' , sorted_dist)         #vara
        # b = sorted_dist[0,0]
        # s = sorted_dist[0,1]
        return sorted_dist[0,:10]

    def KNearest_test(self, X, Y):                   #vara
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
            # y = Y[i]
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
'''
from __future__ import division

import logging

import copy
import numpy as np
import networkx as nx
import scipy.spatial.distance as sp
import matplotlib.pyplot as plt

# TODO should have a _get_activation_trajectories method too

class gwr():


    # Growing When Required (GWR) Neural Gas, after [1]. Constitutes the base class
    # for the Online Semi-supervised (OSS) GWR.

    # [1] Parisi, G. I., Tani, J., Weber, C., & Wermter, S. (2017).
    # Emergence of multimodal action representations from neural network
    # self-organization. Cognitive Systems Research, 43, 208-221.


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

    def _initialize(self, X):

        logging.info('Initializing the neural gas.')
        self.G = nx.Graph()
        # TODO: initialize empty labels?
        draw = np.random.choice(X.shape[0], size=2, replace=False)
        # print(X[draw[0]].shape)
        # print(draw)
        self.G.add_nodes_from([(0,{'pos' : X[draw[0],:],                             'fir' : 1})])      #vara
        self.G.add_nodes_from([(1,{'pos' : X[draw[1],:],
                                     'fir' : 1})])      #vara
        #print(self.G.number_of_nodes())                            #vara
        #print(nx.get_node_attributes(self.G, 'pos').values())      #vara
        #print(self.G.nodes[0])                                     #vara

    def get_positions(self):
        pos = np.array(list(nx.get_node_attributes(self.G, 'pos').values()))    #vara
        return pos

    def _get_best_matching(self, x):
        pos = self.get_positions()
        #np.concatenate(pos,axis = 0)               #vara
        #print(pos)                                 #vara
        dist = sp.cdist(x, pos, metric='euclidean')
        sorted_dist = np.argsort(dist)
        #print('sroted dist' , sorted_dist)         #vara
        b = sorted_dist[0,0]
        s = sorted_dist[0,1]
        return b, s


    def _get_activation(self, x, b):
        p = self.G.nodes[b]['pos'][np.newaxis,:]
        dist = sp.cdist(x, p, metric='euclidean')[0,0]
        act = np.exp(-dist)
        return act


    def _make_link(self, b, s):
        self.G.add_edge(b,s,age = 0)            #vara
        #print('edges ',self.G.edges([0]))      #vara


    def _add_node(self, x, b, s):
        r = max(self.G.nodes()) + 1
        pos_r = 0.5 * (x + self.G.nodes[b]['pos'])[0,:]
        self.G.add_nodes_from([(r, {'pos' : pos_r, 'fir' : 1})])    #vara
        self.G.remove_edge(b,s)
        self.G.add_edge(r, b, age = 0)          #vara
        self.G.add_edge(r, s, age = 0)          #vara
        return r


    def _update_network(self, x, b):
        dpos_b = self.eps_b * self.G.nodes[b]['fir']*(x - self.G.nodes[b]['pos'])
        self.G.nodes[b]['pos'] = self.G.nodes[b]['pos'] + dpos_b[0,:]

        neighbors = self.G.neighbors(b)
        for n in neighbors:
            # update the position of the neighbors
            dpos_n = self.eps_n * self.G.nodes[n]['fir'] * (
                     x - self.G.nodes[n]['pos'])
            self.G.nodes[n]['pos'] = self.G.nodes[n]['pos'] + dpos_n[0,:]

            # increase the age of all edges connected to b
            #print('age ', self.G.edges[b,n]['age'])        #vara
            self.G.edges[b,n]['age'] += 1                   #vara


    def _update_firing(self, b):
        dfir_b = self.tau_b * self.kappa*(1-self.G.nodes[b]['fir']) - self.tau_b        #different
        self.G.nodes[b]['fir'] = self.G.nodes[b]['fir']  + dfir_b
        # if(self.G.nodes[b]['fir']<0):
        #     print("negtive problem")
        self.G.nodes[b]['fir'] = np.clip(self.G.nodes[b]['fir'],0,1) #max(0,self.G.nodes[b]['fir']);
        #print('fir ',self.G.nodes[b]['fir'])   #vara
        #print(self.G.nodes)                    #vara

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
        # TODO: implement this
        pass

    def _training_step(self, x):
        # TODO: do not recompute all positions at every iteration
        b, s = self._get_best_matching(x)
        #print('best match ',b,s)            #vara
        self._make_link(b, s)
        act = self._get_activation(x, b)
        fir = self.G.nodes[b]['fir']
        logging.debug('Training step - best matching: %s, %s \n'
                      'Network activation: %s \n'
                      'Firing: %s', str(b), str(s), str(np.round(act,3)),
                      str(np.round(fir,3)))
        if act < self.act_thr and fir < self.fir_thr \
            and len(self.G.nodes()) < self.max_size:
            r = self._add_node(x, b, s)
            logging.debug('GENERATE NODE %s', self.G.nodes[r])
        else:
            self._update_network(x, b)
        self._update_firing(b)
        self._remove_old_edges()


    def train(self, X, n_epochs=20, warm_start = False):
        if not warm_start:
            self._initialize(X)
        for n in range(n_epochs):
            print('epoch: ',n)
            logging.info('>>> Training epoch %s', str(n))
            for i in range(X.shape[0]):
                x = X[i,np.newaxis]
                self._training_step(x)
                self._check_stopping_criterion()
        return self.G
        #nx.draw(self.G)        #vara
        #plt.show()
        logging.info('Training ended - Network size: %s', len(self.G.nodes()))
'''