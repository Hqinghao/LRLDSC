import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.contrib import layers
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import f1_score
from munkres import Munkres
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from datetime import datetime
import time
import math
import os,traceback
import pandas as pd
from multiprocessing import cpu_count
from sklearn.manifold import TSNE
# SELECT GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.compat.v1.disable_eager_execution()


class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_const1=1.0, reg_const2=1.0, reg_const3=1.0, reg_const4=1.0, reg=None, batch_size=256,
                 ds=None, denoise=False, model_path=None, logs_path='./pretrain/logs'):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.reg = reg
        self.model_path = model_path
        self.kernel_size = kernel_size
        self.iter = 0
        self.batch_size = batch_size
        self.x = tf.compat.v1.placeholder(
            tf.float32, [None, self.n_input[0], self.n_input[1], 1])
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, [])
        weights = self._initialize_weights()

        if denoise == False:
            x_input = self.x
            latent, shape = self.encoder(x_input, weights)
        else:
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                                      mean=0,
                                                      stddev=0.2,
                                                      dtype=tf.float32))
            latent, shape = self.encoder(x_input, weights)
        hid_dim =tf.compat.v1.dimension_value(latent.shape[1]) * tf.compat.v1.dimension_value(latent.shape[2]) * tf.compat.v1.dimension_value(latent.shape[3])
        self.z_conv = tf.reshape(latent, [batch_size, hid_dim])
        Coef = weights['Coef']         
        self.z_ssc = tf.matmul(Coef,self.z_conv)  
        self.Coef = Coef       
        latent_c = tf.reshape(self.z_ssc, tf.shape(latent)) 
  

    	# classifier module
        if ds is not None:
            pslb = tf.keras.layers.Dense(ds,activation="softmax",name = 'ss_d')(self.z_conv)
            cluster_assignment = tf.argmax(pslb, -1)
            
        self.x_r_ft = self.decoder(latent_c, weights, shape)
        # Reconstruction loss
        self.recon = 0.5 * tf.reduce_sum(
            tf.pow(tf.subtract(self.x_r_ft, self.x), 2.0))
        
        # Maximum Entropy (ME) regularization loss, referenced from [Maximum Entropy Subspace Clustering Network/Zhihao Peng]
        self.reg_ssc = 0.5*tf.reduce_sum(tf.multiply((self.Coef), tf.math.log(
            tf.compat.v1.clip_by_value(Coef, clip_value_min=1.0e-12, clip_value_max=1.0))))

        # Self-representation loss
        self.cost_ssc = 0.5 * \
            tf.reduce_sum(tf.pow(tf.subtract(self.z_conv, self.z_ssc), 2))
        #thershold  labelloss
        onesl=np.ones(batch_size)
        zerosl=np.zeros(batch_size)
        weight_label=tf.where(tf.reduce_max(pslb,axis=1)>0.8,onesl,zerosl)
        cluster_assignment1=tf.one_hot(cluster_assignment,ds)
        self.w_weight=weight_label
        self.labelloss=tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=cluster_assignment1,logits=pslb,weights=weight_label)
        #  negative
        similar_martix_sotfmax = cosinem(pslb)
        self.similar_martix_sotfmax=similar_martix_sotfmax
        # th1
        unsimilar_th = 0.1
        us_ones = tf.multiply(-1.0,tf.ones([batch_size,batch_size],dtype='float32'))
        m_zeros = tf.zeros([batch_size,batch_size],dtype='float32')
        similar_martix_n = tf.where(similar_martix_sotfmax <= unsimilar_th,us_ones,m_zeros)
        # positive
        similar_martix_selfexspress = Coef
        self.similar_martix_selfexspress=similar_martix_selfexspress
        similar_martix_p = compute_similar_martix_p(similar_martix_selfexspress)
        # negative + positive 
        similar_martix = tf.add(similar_martix_n,similar_martix_p)
        self.similar_martix=similar_martix
        #LA
        S = tf.multiply(0.5,tf.add(similar_martix,tf.transpose(similar_martix)))
        D = tf.linalg.diag(tf.sqrt((1.0 / tf.reduce_sum(tf.abs(S), axis=1))))
        La = tf.eye(tf.shape(S)[0],dtype=tf.float32) - tf.matmul(tf.matmul(D,S),D)
        self.regular_z = tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(self.z_conv),tf.cast(La,tf.float32)), self.z_conv))

        
        tf.compat.v1.summary.scalar("self_expressive_loss", self.cost_ssc)
        tf.compat.v1.summary.scalar("coefficient_loss", self.reg_ssc)
        tf.compat.v1.summary.scalar("reconstruction loss", self.recon)
        self.loss_ssc = self.recon + reg_const1 * self.reg_ssc + reg_const2 * self.cost_ssc + reg_const3 * self.regular_z + reg_const4 * self.labelloss
        

        self.merged_summary_op = tf.compat.v1.summary.merge_all()
        self.optimizer_ssc = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss_ssc)
        self.init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.InteractiveSession()
        self.summary_writer = tf.compat.v1.summary.FileWriter(
            logs_path, graph=tf.compat.v1.get_default_graph())
        self.saver = tf.compat.v1.train.Saver(
            [v for v in tf.compat.v1.trainable_variables() if not (v.name.startswith("Coef")or v.name.startswith("ss"))])

        self.sess.run(self.init)

    def _initialize_weights(self):
        all_weights = dict()
        n_layers = len(self.n_hidden)
        all_weights['Coef']   = tf.compat.v1.Variable(1.0e-5 * tf.ones([self.batch_size, self.batch_size],tf.float32), name = 'Coef')        
        all_weights['enc_w0'] = tf.compat.v1.get_variable("enc_w0", shape=[
                                                          self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]], initializer=tf.compat.v1.keras.initializers.he_normal(), regularizer=self.reg)
        all_weights['enc_b0'] = tf.compat.v1.Variable(
            tf.zeros([self.n_hidden[0]], dtype=tf.float32))
        iter_i = 1
        while iter_i < n_layers:
            enc_name_wi = 'enc_w' + str(iter_i)
            all_weights[enc_name_wi] = tf.compat.v1.get_variable(enc_name_wi, shape=[self.kernel_size[iter_i], self.kernel_size[iter_i],
                                                                                     self.n_hidden[iter_i-1], self.n_hidden[iter_i]], initializer=tf.compat.v1.keras.initializers.he_normal(), regularizer=self.reg)
            enc_name_bi = 'enc_b' + str(iter_i)
            all_weights[enc_name_bi] = tf.compat.v1.Variable(
                tf.zeros([self.n_hidden[iter_i]], dtype=tf.float32))
            iter_i = iter_i + 1
        iter_i = 1
        while iter_i < n_layers:
            dec_name_wi = 'dec_w' + str(iter_i - 1)
            all_weights[dec_name_wi] = tf.compat.v1.get_variable(dec_name_wi, shape=[self.kernel_size[n_layers-iter_i], self.kernel_size[n_layers-iter_i],
                                                                                     self.n_hidden[n_layers-iter_i-1], self.n_hidden[n_layers-iter_i]], initializer=tf.compat.v1.keras.initializers.he_normal(), regularizer=self.reg)
            dec_name_bi = 'dec_b' + str(iter_i - 1)
            all_weights[dec_name_bi] = tf.compat.v1.Variable(
                tf.zeros([self.n_hidden[n_layers-iter_i-1]], dtype=tf.float32))
            iter_i = iter_i + 1
        dec_name_wi = 'dec_w' + str(iter_i - 1)
        all_weights[dec_name_wi] = tf.compat.v1.get_variable(dec_name_wi, shape=[
                                                             self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]], initializer=tf.compat.v1.keras.initializers.he_normal(), regularizer=self.reg)
        dec_name_bi = 'dec_b' + str(iter_i - 1)
        all_weights[dec_name_bi] = tf.compat.v1.Variable(
            tf.zeros([1], dtype=tf.float32))
        return all_weights

    # Encoder
    def encoder(self, x, weights):
        shapes = []
        shapes.append(x.get_shape().as_list())
        layeri = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[
                                1, 2, 2, 1], padding='SAME'), weights['enc_b0'])
        layeri = tf.nn.relu(layeri)
        shapes.append(layeri.get_shape().as_list())
        n_layers = len(self.n_hidden)
        iter_i = 1
        while iter_i < n_layers:
            layeri = tf.nn.bias_add(tf.nn.conv2d(layeri, weights['enc_w' + str(iter_i)], strides=[
                                    1, 2, 2, 1], padding='SAME'), weights['enc_b' + str(iter_i)])
            layeri = tf.nn.relu(layeri)
            shapes.append(layeri.get_shape().as_list())
            iter_i = iter_i + 1
        layer3 = layeri
        return layer3, shapes

    # Decoder
    def decoder(self, z, weights, shapes):
        n_layers = len(self.n_hidden)
        layer3 = z
        iter_i = 0
        while iter_i < n_layers:
            shape_de = shapes[n_layers - iter_i - 1]
            layer3 = tf.add(tf.nn.conv2d_transpose(layer3, weights['dec_w' + str(iter_i)], tf.stack([tf.shape(self.x)[
                            0], shape_de[1], shape_de[2], shape_de[3]]), strides=[1, 2, 2, 1], padding='SAME'), weights['dec_b' + str(iter_i)])
            layer3 = tf.nn.relu(layer3)
            iter_i = iter_i + 1
        return layer3

    def finetune_fit(self, X, lr):
        C, l_cost, l1_cost, l2_cost,summary, _ = self.sess.run(
            (self.Coef, self.loss_ssc, self.reg_ssc, self.cost_ssc,self.merged_summary_op, self.optimizer_ssc), feed_dict={self.x: X, self.learning_rate: lr})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return C, l_cost, l1_cost, l2_cost

    def initlization(self):
        self.sess.run(self.init)

    # For the close of interactive session
    def runclose(self):
        self.sess.close()
        print("InteractiveSession.close()")

    def restore(self):
        self.saver.restore(self.sess, self.model_path)
        print("model restored")


# L1: Groundtruth labels; L2: Clustering labels;
def best_map(L1, L2):
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t, i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1, i], i] = C[Ind[0:t+1, i], i]
                t = t + 1
    else:
        Cp = C
    return Cp

# C: coefficient matrix; K: number of clusters; d: dimension of each subspace;
def post_proC(C, K, d, alpha):
    C = 0.5*(C + C.T)
    n = C.shape[0]
    C = C - np.diag(np.diag(C)) + np.eye(n, n)
    r = min(d*K + 1, C.shape[0]-1)      
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L/L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(
        n_clusters=K, eigen_solver='arpack', affinity='precomputed', assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L

def err_rate(gt_s, s):
    c_x = best_map(gt_s,s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    NMI = metrics.normalized_mutual_info_score(gt_s, c_x)

    purity = 0
    N = gt_s.shape[0]
    Label1 = np.unique(gt_s)
    nClass1 = len(Label1)
    Label2 = np.unique(c_x)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    for label in Label2:
        tempc = [i for i in range(N) if s[i] == label]
        hist,bin_edges = np.histogram(gt_s[tempc],Label1)
        purity += max([np.max(hist),len(tempc)-np.sum(hist)])
    purity /= N
    return missrate,NMI,purity

def cosinem(X):
    normalized = tf.nn.l2_normalize(X,axis=1)
    cosine_s = tf.matmul(normalized,tf.transpose(normalized))
    return cosine_s

def compute_similar_martix_p(X):
    As = tf.multiply(0.5,tf.add(tf.abs(X),tf.abs(tf.transpose(X))))
    mask = tf.eye(tf.shape(X)[0], dtype=tf.bool)
    si_confi = tf.ones_like(As)
    si_unconfi = tf.zeros_like(As)
    similar_martix_p = tf.where(As > tf.reduce_mean(As), si_confi, si_unconfi)
    similar_martix_p = tf.where(mask, tf.ones_like(similar_martix_p), similar_martix_p)
    return similar_martix_p

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orangered','greenyellow','darkviolet']
marks = ['o','+','.']
def visualize(Img,Label,filep=None):
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax1 = fig.add_subplot(111)
    n = Img.shape[0]
    Z_emb = TSNE(n_components=2, metric='precomputed').fit_transform(Img, Label)
    # print(Z_emb)
    lbs = np.unique(Label)
    for ii in range(lbs.size):
        Z_embi = Z_emb[[i for i in range(n) if Label[i] == lbs[ii]]].transpose()
        # print(Z_embi)
        ax1.scatter(Z_embi[0], Z_embi[1], color=colors[ii % 10], marker=marks[ii // 10], label=str(ii),s=3)
    ax1.legend()
    if filep is not None:
        plt.savefig(filep)
    plt.show()



if __name__ == '__main__':
    
    # load face images and labels
    data = sio.loadmat('./Data/COIL20.mat')
    Img = data['fea']
    print(Img[:5,:3])
    Label = data['gnd']  
    
    # face image clustering
    n_input = [32,32]
    kernel_size = [3]
    n_hidden = [15]
    
    Img = np.reshape(Img,[Img.shape[0],n_input[0],n_input[1],1]) 
    print(Img.shape)
    all_subjects = [20]
    mm = 0
    ro = 0.03
    lr = 1e-4
    epoch = 0
    num_class = all_subjects[0]
    batch_size = num_class* 72
    max_step = 200#50 + num_class*25# 100+num_class*20
    display_step = 2#10
    mreg=[0,0,0,0]
    
    # max [75, 50, 50, 70, 0.0001] 0.9854166666666667
    

    model_path = './pre_train-tf2/COIL20/model-COIL20.ckpt'
    logs_path = './fit_model/conv_3_l2_other/ft/logs'
    for reg1 in [0.01,0.1,1,10,100]:
        for reg2 in [0.01,0.1,1,10,100]:
            for reg3 in[0.01,0.1,1,10,100]:
                for reg4 in [0.01,0.1,1,10,100]:
                    try:
                        print('reg',reg1,reg2,reg3,reg4)
                        parameters = [reg1,reg2,reg3,reg4,lr]
                        avg = []
                        med = []
                        tf.compat.v1.reset_default_graph()
                        #CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1, re_constant2=reg2, \
                        #            kernel_size=kernel_size, batch_size=batch_size, model_path=model_path, restore_path=restore_path, logs_path=logs_path)
                        CAE = ConvAE( n_input=n_input, n_hidden=n_hidden, reg_const1=reg1, reg_const2=reg2,reg_const3=reg3,reg_const4=reg4,ds=num_class, \
                                     kernel_size=kernel_size, batch_size=batch_size, model_path=model_path,logs_path=logs_path)
                        acc_= []
                        for i in range(0,1):
                            sample_all_subjs = Img
                            sample_all_subjs = sample_all_subjs.astype(float)
                            label_all_subjs = Label
                            label_all_subjs = label_all_subjs - label_all_subjs.min() + 1
                            label_all_subjs = np.squeeze(label_all_subjs)
                            CAE.initlization()
                            CAE.restore()	
                            epoch_list=[]
                            cost_list=[]   
                            acc_list=[]
                            nmi_list=[]
                            purity_list=[]

                            while epoch < max_step:
                                epoch = epoch + 1         
                                C,l_cost,l1_cost,l2_cost = CAE.finetune_fit(sample_all_subjs,lr)  
                                epoch_list.append(epoch)
                                cost_list.append(round(l_cost/float(batch_size),3))                           
                                if  epoch>30:
                                    print ("epoch: %.1d" % epoch, "cost: %.8f" % (l_cost/float(batch_size)))                
                                    C = thrC(C,ro)                                                       
                                    y_x, La = post_proC(C,num_class, 16,6)                 
                                    missrate_x,NMI,purity = err_rate(label_all_subjs, y_x)                
                                    acc_x = 1 - missrate_x 
                                    acc_list.append(acc_x)
                                    nmi_list.append(NMI)
                                    purity_list.append(purity)
                                    print ("epoch: %d" % epoch, "our accuracy: %.4f" % acc_x)
                                    acc_.append(acc_x)  
                                    save_info = 'parameters: ' + str(parameters) + ': ' + 'epoch'+':' + str(epoch) + '------accury: ' + str(acc_x) + ': ' + 'NMI: ' + str(NMI) + ': ' + 'purity: ' +str(purity) +  '\n'
                                    with open('result-COIL20','a+') as f:
                                        f.write(save_info)
                                        
                            save_info_max = 'parameters: ' + str(parameters) + 'max_info: ' + 'accury: ' + str(max(acc_list)) + ': ' + 'NMI: ' + str(max(nmi_list)) + ': ' + 'purity: ' +str(max(purity_list)) +  '\n'
                            with open('result-COIL20','a+') as f:
                                f.write(save_info_max)
                            
                            acc_ = np.array(acc_)      
                            if max(acc_) > mm:
                                mm = max(acc_)
                                mreg = [reg1,reg2,reg3,reg4,lr]
                            print('max',mreg,mm)                            	                            	                            
                    except:
                        traceback.print_exc()
                    finally:
                        epoch = 0
                        try:
                            CAE.runclose()
                        except:
                            ''
