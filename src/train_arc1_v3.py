import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy
import theano
import theano.tensor as T
import theano.sandbox.neighbours as TSN
import time

from logistic_sgd import LogisticRegression
from WPDefined import ConvFoldPoolLayer, dropout_from_layer, shared_dataset, repeat_whole_matrix
from cis.deep.utils.theano import debug_print
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from loadData import load_MCTest_corpus_DQAAAA, load_word2vec_to_init
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import GRU_Tensor3_Input, GRU_Matrix_Input, Average_Pooling_for_Top, create_GRU_para, Average_Pooling, create_highw_para, Average_Pooling_Scan, create_HiddenLayer_para
from random import shuffle
from mlp import HiddenLayer

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy import linalg, mat, dot

#from preprocess_wikiQA import compute_map_mrr

#need to change
'''
1) add linguistics features
3) dropout
5) add sent-level, doc-level and overall-level similarity together to ranking loss
6) shuffle training data
7) attention used cosine
8) reduce kern, emb size for overfitting



Doesnt work:
3) margin=0.5
4) euclidean distance
4) glove initialization
2) unknown words have different random vectors

'''

def evaluate_lenet5(learning_rate=0.05, n_epochs=2000, nkerns=[50,50], batch_size=1, window_width=3,
                    maxSentLength=64, maxDocLength=60, emb_size=50, hidden_size=200,
                    L2_weight=0.0065, update_freq=1, norm_threshold=5.0, max_s_length=57, max_d_length=59, margin=1.0, decay=0.95):
    maxSentLength=max_s_length+2*(window_width-1)
    maxDocLength=max_d_length+2*(window_width-1)
    model_options = locals().copy()
    print "model options", model_options
    rootPath='/mounts/data/proj/wenpeng/Dataset/MCTest/';
    rng = numpy.random.RandomState(23455)
    train_data,train_size, test_data, test_size, vocab_size=load_MCTest_corpus_DQAAAA(rootPath+'vocab_DQAAAA.txt', rootPath+'mc500.train.tsv_standardlized.txt_DQAAAA.txt', rootPath+'mc500.test.tsv_standardlized.txt_DQAAAA.txt', max_s_length,maxSentLength, maxDocLength)#vocab_size contain train, dev and test


    [train_data_D, train_data_Q, train_data_A1, train_data_A2, train_data_A3, train_data_A4, train_Label, 
                 train_Length_D,train_Length_D_s, train_Length_Q, train_Length_A1, train_Length_A2, train_Length_A3, train_Length_A4,
                train_leftPad_D,train_leftPad_D_s, train_leftPad_Q, train_leftPad_A1, train_leftPad_A2, train_leftPad_A3, train_leftPad_A4,
                train_rightPad_D,train_rightPad_D_s, train_rightPad_Q, train_rightPad_A1, train_rightPad_A2, train_rightPad_A3, train_rightPad_A4]=train_data
    [test_data_D, test_data_Q, test_data_A1, test_data_A2, test_data_A3, test_data_A4, test_Label, 
                 test_Length_D,test_Length_D_s, test_Length_Q, test_Length_A1, test_Length_A2, test_Length_A3, test_Length_A4,
                test_leftPad_D,test_leftPad_D_s, test_leftPad_Q, test_leftPad_A1, test_leftPad_A2, test_leftPad_A3, test_leftPad_A4,
                test_rightPad_D,test_rightPad_D_s, test_rightPad_Q, test_rightPad_A1, test_rightPad_A2, test_rightPad_A3, test_rightPad_A4]=test_data                


    n_train_batches=train_size/batch_size
    n_test_batches=test_size/batch_size
    
    train_batch_start=list(numpy.arange(n_train_batches)*batch_size)
    test_batch_start=list(numpy.arange(n_test_batches)*batch_size)

    
#     indices_train_l=theano.shared(numpy.asarray(indices_train_l, dtype=theano.config.floatX), borrow=True)
#     indices_train_r=theano.shared(numpy.asarray(indices_train_r, dtype=theano.config.floatX), borrow=True)
#     indices_test_l=theano.shared(numpy.asarray(indices_test_l, dtype=theano.config.floatX), borrow=True)
#     indices_test_r=theano.shared(numpy.asarray(indices_test_r, dtype=theano.config.floatX), borrow=True)
#     indices_train_l=T.cast(indices_train_l, 'int64')
#     indices_train_r=T.cast(indices_train_r, 'int64')
#     indices_test_l=T.cast(indices_test_l, 'int64')
#     indices_test_r=T.cast(indices_test_r, 'int64')
    


    rand_values=random_value_normal((vocab_size+1, emb_size), theano.config.floatX, numpy.random.RandomState(1234))
    rand_values[0]=numpy.array(numpy.zeros(emb_size),dtype=theano.config.floatX)
    #rand_values[0]=numpy.array([1e-50]*emb_size)
    rand_values=load_word2vec_to_init(rand_values, rootPath+'vocab_DQAAAA_glove_50d.txt')
    #rand_values=load_word2vec_to_init(rand_values, rootPath+'vocab_lower_in_word2vec_embs_300d.txt')
    embeddings=theano.shared(value=rand_values, borrow=True)      
    
    #cost_tmp=0
    error_sum=0
    
    # allocate symbolic variables for the data
    index = T.lscalar()
    index_D = T.lmatrix()   # now, x is the index matrix, must be integer
    index_Q = T.lvector()
    index_A1= T.lvector()
    index_A2= T.lvector()
    index_A3= T.lvector()
    index_A4= T.lvector()
#     y = T.lvector()  
    
    len_D=T.lscalar()
    len_D_s=T.lvector()
    len_Q=T.lscalar()
    len_A1=T.lscalar()
    len_A2=T.lscalar()
    len_A3=T.lscalar()
    len_A4=T.lscalar()

    left_D=T.lscalar()
    left_D_s=T.lvector()
    left_Q=T.lscalar()
    left_A1=T.lscalar()
    left_A2=T.lscalar()
    left_A3=T.lscalar()
    left_A4=T.lscalar()

    right_D=T.lscalar()
    right_D_s=T.lvector()
    right_Q=T.lscalar()
    right_A1=T.lscalar()
    right_A2=T.lscalar()
    right_A3=T.lscalar()
    right_A4=T.lscalar()
        


    #x=embeddings[x_index.flatten()].reshape(((batch_size*4),maxSentLength, emb_size)).transpose(0, 2, 1).flatten()
    ishape = (emb_size, maxSentLength)  # sentence shape
    dshape = (nkerns[0], maxDocLength) # doc shape
    filter_words=(emb_size,window_width)
    filter_sents=(nkerns[0], window_width)
    #poolsize1=(1, ishape[1]-filter_size[1]+1) #?????????????????????????????
#     length_after_wideConv=ishape[1]+filter_size[1]-1
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    #layer0_input = x.reshape(((batch_size*4), 1, ishape[0], ishape[1]))
    layer0_D_input = debug_print(embeddings[index_D.flatten()].reshape((maxDocLength,maxSentLength, emb_size)).transpose(0, 2, 1), 'layer0_D_input')#.dimshuffle(0, 'x', 1, 2)
    layer0_Q_input = debug_print(embeddings[index_Q.flatten()].reshape((maxSentLength, emb_size)).transpose(), 'layer0_Q_input')#.dimshuffle(0, 'x', 1, 2)
    layer0_A1_input = debug_print(embeddings[index_A1.flatten()].reshape((maxSentLength, emb_size)).transpose(), 'layer0_A1_input')#.dimshuffle(0, 'x', 1, 2)
    layer0_A2_input = embeddings[index_A2.flatten()].reshape((maxSentLength, emb_size)).transpose()#.dimshuffle(0, 'x', 1, 2)
    layer0_A3_input = embeddings[index_A3.flatten()].reshape((maxSentLength, emb_size)).transpose()#.dimshuffle(0, 'x', 1, 2)
    layer0_A4_input = embeddings[index_A4.flatten()].reshape((maxSentLength, emb_size)).transpose()#.dimshuffle(0, 'x', 1, 2)
    
        
    U, W, b=create_GRU_para(rng, emb_size, nkerns[0])
    layer0_para=[U, W, b] 
#     conv2_W, conv2_b=create_conv_para(rng, filter_shape=(nkerns[1], 1, nkerns[0], filter_sents[1]))
#     layer2_para=[conv2_W, conv2_b]
#     high_W, high_b=create_highw_para(rng, nkerns[0], nkerns[1])
#     highW_para=[high_W, high_b]

    #load_model(params)
    
    
    layer0_D = GRU_Tensor3_Input(T=layer0_D_input[left_D:-right_D,:,:],
                                 lefts=left_D_s[left_D:-right_D],
                                 rights=right_D_s[left_D:-right_D],
                                 hidden_dim=nkerns[0],
                                 U=U,W=W,b=b)
    layer0_Q = GRU_Matrix_Input(X=layer0_Q_input[:,left_Q:-right_Q], word_dim=emb_size, hidden_dim=nkerns[0],U=U,W=W,b=b,bptt_truncate=-1)
    layer0_A1 = GRU_Matrix_Input(X=layer0_A1_input[:,left_A1:-right_A1], word_dim=emb_size, hidden_dim=nkerns[0],U=U,W=W,b=b,bptt_truncate=-1)
    layer0_A2 = GRU_Matrix_Input(X=layer0_A2_input[:,left_A2:-right_A2], word_dim=emb_size, hidden_dim=nkerns[0],U=U,W=W,b=b,bptt_truncate=-1)
    layer0_A3 = GRU_Matrix_Input(X=layer0_A3_input[:,left_A3:-right_A3], word_dim=emb_size, hidden_dim=nkerns[0],U=U,W=W,b=b,bptt_truncate=-1)
    layer0_A4 = GRU_Matrix_Input(X=layer0_A4_input[:,left_A4:-right_A4], word_dim=emb_size, hidden_dim=nkerns[0],U=U,W=W,b=b,bptt_truncate=-1)

    
    layer0_D_output=debug_print(layer0_D.output, 'layer0_D.output')
    layer0_Q_output=debug_print(layer0_Q.output_vector_mean, 'layer0_Q.output')
    layer0_A1_output=debug_print(layer0_A1.output_vector_mean, 'layer0_A1.output')
    layer0_A2_output=debug_print(layer0_A2.output_vector_mean, 'layer0_A2.output')
    layer0_A3_output=debug_print(layer0_A3.output_vector_mean, 'layer0_A3.output')
    layer0_A4_output=debug_print(layer0_A4.output_vector_mean, 'layer0_A4.output')
    
    #before reasoning, do a GRU for doc: d
    U_d, W_d, b_d=create_GRU_para(rng, nkerns[0], nkerns[0])
    layer_d_para=[U_d, W_d, b_d]
    layer_D_GRU = GRU_Matrix_Input(X=layer0_D_output, word_dim=nkerns[0], hidden_dim=nkerns[0],U=U_d,W=W_d,b=b_d,bptt_truncate=-1)
    
    #Reasoning Layer 1
    repeat_Q=debug_print(T.repeat(layer0_Q_output.reshape((layer0_Q_output.shape[0],1)), maxDocLength, axis=1)[:,:layer_D_GRU.output_matrix.shape[1]], 'repeat_Q')
    input_DNN=debug_print(T.concatenate([layer_D_GRU.output_matrix,repeat_Q], axis=0).transpose(), 'input_DNN')#each row is an example
    output_DNN1=HiddenLayer(rng, input=input_DNN, n_in=nkerns[0]*2, n_out=nkerns[0])
    output_DNN2=HiddenLayer(rng, input=output_DNN1.output, n_in=nkerns[0], n_out=nkerns[0])
    
    DNN_out=debug_print(output_DNN2.output.transpose(), 'DNN_out')
    U_p, W_p, b_p=create_GRU_para(rng, nkerns[0], nkerns[0])
    layer_pooling_para=[U_p, W_p, b_p] 
    pooling=GRU_Matrix_Input(X=DNN_out, word_dim=nkerns[0], hidden_dim=nkerns[0],U=U_p,W=W_p,b=b_p,bptt_truncate=-1)
    translated_Q1=debug_print(pooling.output_vector_max, 'translated_Q1')


    #before reasoning, do a GRU for doc: d2
    U_d2, W_d2, b_d2=create_GRU_para(rng, nkerns[0], nkerns[0])
    layer_d2_para=[U_d2, W_d2, b_d2]
    layer_D2_GRU = GRU_Matrix_Input(X=layer_D_GRU.output_matrix, word_dim=nkerns[0], hidden_dim=nkerns[0],U=U_d2,W=W_d2,b=b_d2,bptt_truncate=-1)
    #Reasoning Layer 2
    repeat_Q1=debug_print(T.repeat(translated_Q1.reshape((translated_Q1.shape[0],1)), maxDocLength, axis=1)[:,:layer_D2_GRU.output_matrix.shape[1]], 'repeat_Q1')
    input_DNN2=debug_print(T.concatenate([layer_D2_GRU.output_matrix,repeat_Q1], axis=0).transpose(), 'input_DNN2')#each row is an example
    output_DNN3=HiddenLayer(rng, input=input_DNN2, n_in=nkerns[0]*2, n_out=nkerns[0])
    output_DNN4=HiddenLayer(rng, input=output_DNN3.output, n_in=nkerns[0], n_out=nkerns[0])
    
    DNN_out2=debug_print(output_DNN4.output.transpose(), 'DNN_out2')
    U_p2, W_p2, b_p2=create_GRU_para(rng, nkerns[0], nkerns[0])
    layer_pooling_para2=[U_p2, W_p2, b_p2] 
    pooling2=GRU_Matrix_Input(X=DNN_out2, word_dim=nkerns[0], hidden_dim=nkerns[0],U=U_p2,W=W_p2,b=b_p2,bptt_truncate=-1)
    translated_Q2=debug_print(pooling2.output_vector_max, 'translated_Q2')
    

    QA1=T.concatenate([translated_Q2, layer0_A1_output], axis=0)
    QA2=T.concatenate([translated_Q2, layer0_A2_output], axis=0)
    QA3=T.concatenate([translated_Q2, layer0_A3_output], axis=0)
    QA4=T.concatenate([translated_Q2, layer0_A4_output], axis=0)
    
    W_HL,b_HL=create_HiddenLayer_para(rng, n_in=nkerns[0]*2, n_out=1)
    match_params=[W_HL,b_HL]
    QA1_match=HiddenLayer(rng, input=QA1, n_in=nkerns[0]*2, n_out=1, W=W_HL, b=b_HL)
    QA2_match=HiddenLayer(rng, input=QA2, n_in=nkerns[0]*2, n_out=1, W=W_HL, b=b_HL)
    QA3_match=HiddenLayer(rng, input=QA3, n_in=nkerns[0]*2, n_out=1, W=W_HL, b=b_HL)
    QA4_match=HiddenLayer(rng, input=QA4, n_in=nkerns[0]*2, n_out=1, W=W_HL, b=b_HL)
    
    
    
#     simi_overall_level1=debug_print(cosine(translated_Q2, layer0_A1_output), 'simi_overall_level1')
#     simi_overall_level2=debug_print(cosine(translated_Q2, layer0_A2_output), 'simi_overall_level2')
#     simi_overall_level3=debug_print(cosine(translated_Q2, layer0_A3_output), 'simi_overall_level3')
#     simi_overall_level4=debug_print(cosine(translated_Q2, layer0_A4_output), 'simi_overall_level4')

    simi_overall_level1=debug_print(QA1_match.output[0], 'simi_overall_level1')
    simi_overall_level2=debug_print(QA2_match.output[0], 'simi_overall_level2')
    simi_overall_level3=debug_print(QA3_match.output[0], 'simi_overall_level3')
    simi_overall_level4=debug_print(QA4_match.output[0], 'simi_overall_level4')


#     eucli_1=1.0/(1.0+EUCLID(layer3_DQ.output_D+layer3_DA.output_D, layer3_DQ.output_QA+layer3_DA.output_QA))
 
    #only use overall_simi    
    cost=T.maximum(0.0, margin+simi_overall_level2-simi_overall_level1)+T.maximum(0.0, margin+simi_overall_level3-simi_overall_level1)+T.maximum(0.0, margin+simi_overall_level4-simi_overall_level1)
    
#     cost=T.maximum(0.0, margin+T.max([simi_overall_level2, simi_overall_level3, simi_overall_level4])-simi_overall_level1) # ranking loss: max(0, margin-nega+posi)
    posi_simi=simi_overall_level1
    nega_simi=T.max([simi_overall_level2, simi_overall_level3, simi_overall_level4])
#     #use ensembled simi
#     cost=T.maximum(0.0, margin+T.max([simi_2, simi_3, simi_4])-simi_1) # ranking loss: max(0, margin-nega+posi)
#     posi_simi=simi_1
#     nega_simi=T.max([simi_2, simi_3, simi_4])


    
    L2_reg =debug_print((U**2).sum()+(W**2).sum()
                        +(U_p**2).sum()+(W_p**2).sum()
                        +(U_p2**2).sum()+(W_p2**2).sum()
                        +(U_d**2).sum()+(W_d**2).sum()
                        +(U_d2**2).sum()+(W_d2**2).sum()
                        +(output_DNN1.W**2).sum()+(output_DNN2.W**2).sum()
                        +(output_DNN3.W**2).sum()+(output_DNN4.W**2).sum()
                        +(W_HL**2).sum(), 'L2_reg')#+(embeddings**2).sum(), 'L2_reg')#+(layer1.W** 2).sum()++(embeddings**2).sum()
    cost=debug_print(cost+L2_weight*L2_reg, 'cost')
    #cost=debug_print((cost_this+cost_tmp)/update_freq, 'cost')
    


    
    test_model = theano.function([index], [cost, posi_simi, nega_simi],
          givens={
            index_D: test_data_D[index], #a matrix
            index_Q: test_data_Q[index],
            index_A1: test_data_A1[index],
            index_A2: test_data_A2[index],
            index_A3: test_data_A3[index],
            index_A4: test_data_A4[index],

            len_D: test_Length_D[index],
            len_D_s: test_Length_D_s[index],
            len_Q: test_Length_Q[index],
            len_A1: test_Length_A1[index],
            len_A2: test_Length_A2[index],
            len_A3: test_Length_A3[index],
            len_A4: test_Length_A4[index],

            left_D: test_leftPad_D[index],
            left_D_s: test_leftPad_D_s[index],
            left_Q: test_leftPad_Q[index],
            left_A1: test_leftPad_A1[index],
            left_A2: test_leftPad_A2[index],
            left_A3: test_leftPad_A3[index],
            left_A4: test_leftPad_A4[index],
        
            right_D: test_rightPad_D[index],
            right_D_s: test_rightPad_D_s[index],
            right_Q: test_rightPad_Q[index],
            right_A1: test_rightPad_A1[index],
            right_A2: test_rightPad_A2[index],
            right_A3: test_rightPad_A3[index],
            right_A4: test_rightPad_A4[index]
            
            }, on_unused_input='ignore')


    params = layer0_para+output_DNN1.params+output_DNN2.params+output_DNN3.params+output_DNN4.params+layer_pooling_para+layer_pooling_para2+match_params+layer_d_para+layer_d2_para
    
    
#     accumulator=[]
#     for para_i in params:
#         eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
#         accumulator.append(theano.shared(eps_p, borrow=True))
      
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)


#     updates = []
#     for param_i, grad_i, acc_i in zip(params, grads, accumulator):
#         grad_i=debug_print(grad_i,'grad_i')
#         acc = decay*acc_i + (1-decay)*T.sqr(grad_i) #rmsprop
#         updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(acc+1e-6)))   
#         updates.append((acc_i, acc))      
 
    def AdaDelta_updates(parameters,gradients,rho,eps):
        # create variables to store intermediate updates
        gradients_sq = [ theano.shared(numpy.zeros(p.get_value().shape)) for p in parameters ]
        deltas_sq = [ theano.shared(numpy.zeros(p.get_value().shape)) for p in parameters ]
     
        # calculates the new "average" delta for the next iteration
        gradients_sq_new = [ rho*g_sq + (1-rho)*(g**2) for g_sq,g in zip(gradients_sq,gradients) ]
     
        # calculates the step in direction. The square root is an approximation to getting the RMS for the average value
        deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in zip(deltas_sq,gradients_sq_new,gradients) ]
     
        # calculates the new "average" deltas for the next step.
        deltas_sq_new = [ rho*d_sq + (1-rho)*(d**2) for d_sq,d in zip(deltas_sq,deltas) ]
     
        # Prepare it as a list f
        gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
        deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
        parameters_updates = [ (p,p - d) for p,d in zip(parameters,deltas) ]
        return gradient_sq_updates + deltas_sq_updates + parameters_updates   
    
    updates=AdaDelta_updates(params, grads, decay, 1e-6)
  
    train_model = theano.function([index], [cost, posi_simi, nega_simi], updates=updates,
          givens={
            index_D: train_data_D[index],
            index_Q: train_data_Q[index],
            index_A1: train_data_A1[index],
            index_A2: train_data_A2[index],
            index_A3: train_data_A3[index],
            index_A4: train_data_A4[index],

            len_D: train_Length_D[index],
            len_D_s: train_Length_D_s[index],
            len_Q: train_Length_Q[index],
            len_A1: train_Length_A1[index],
            len_A2: train_Length_A2[index],
            len_A3: train_Length_A3[index],
            len_A4: train_Length_A4[index],

            left_D: train_leftPad_D[index],
            left_D_s: train_leftPad_D_s[index],
            left_Q: train_leftPad_Q[index],
            left_A1: train_leftPad_A1[index],
            left_A2: train_leftPad_A2[index],
            left_A3: train_leftPad_A3[index],
            left_A4: train_leftPad_A4[index],
        
            right_D: train_rightPad_D[index],
            right_D_s: train_rightPad_D_s[index],
            right_Q: train_rightPad_Q[index],
            right_A1: train_rightPad_A1[index],
            right_A2: train_rightPad_A2[index],
            right_A3: train_rightPad_A3[index],
            right_A4: train_rightPad_A4[index]
            }, on_unused_input='ignore')

    train_model_predict = theano.function([index], [cost, posi_simi, nega_simi],
          givens={
            index_D: train_data_D[index],
            index_Q: train_data_Q[index],
            index_A1: train_data_A1[index],
            index_A2: train_data_A2[index],
            index_A3: train_data_A3[index],
            index_A4: train_data_A4[index],

            len_D: train_Length_D[index],
            len_D_s: train_Length_D_s[index],
            len_Q: train_Length_Q[index],
            len_A1: train_Length_A1[index],
            len_A2: train_Length_A2[index],
            len_A3: train_Length_A3[index],
            len_A4: train_Length_A4[index],

            left_D: train_leftPad_D[index],
            left_D_s: train_leftPad_D_s[index],
            left_Q: train_leftPad_Q[index],
            left_A1: train_leftPad_A1[index],
            left_A2: train_leftPad_A2[index],
            left_A3: train_leftPad_A3[index],
            left_A4: train_leftPad_A4[index],
        
            right_D: train_rightPad_D[index],
            right_D_s: train_rightPad_D_s[index],
            right_Q: train_rightPad_Q[index],
            right_A1: train_rightPad_A1[index],
            right_A2: train_rightPad_A2[index],
            right_A3: train_rightPad_A3[index],
            right_A4: train_rightPad_A4[index]
            }, on_unused_input='ignore')



    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 500000000000000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    mid_time = start_time

    epoch = 0
    done_looping = False
    
    max_acc=0.0
    best_epoch=0

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #for minibatch_index in xrange(n_train_batches): # each batch
        minibatch_index=0
#         shuffle(train_batch_start)#shuffle training data


        corr_train=0
        for batch_start in train_batch_start: 
            # iter means how many batches have been runed, taking into loop
            iter = (epoch - 1) * n_train_batches + minibatch_index +1
            sys.stdout.write( "Training :[%6f] %% complete!\r" % ((iter%train_size)*100.0/train_size) )
            sys.stdout.flush()

            minibatch_index=minibatch_index+1
            
            cost_average, posi_simi, nega_simi= train_model(batch_start)
            if posi_simi>nega_simi:
                corr_train+=1
            
            if iter % n_train_batches == 0:
                print 'training @ iter = '+str(iter)+' average cost: '+str(cost_average)+'corr rate:'+str(corr_train*100.0/train_size)

            
            if iter % validation_frequency == 0:
                corr_test=0
                for i in test_batch_start:
                    cost, posi_simi, nega_simi=test_model(i)
                    if posi_simi>nega_simi:
                        corr_test+=1

                #write_file.close()
                #test_score = numpy.mean(test_losses)
                test_acc=corr_test*1.0/test_size
                #test_acc=1-test_score
                print(('\t\t\tepoch %i, minibatch %i/%i, test acc of best '
                           'model %f %%') %
                          (epoch, minibatch_index, n_train_batches,test_acc * 100.))
                #now, see the results of LR
                #write_feature=open(rootPath+'feature_check.txt', 'w')
                 

  
                find_better=False
                if test_acc > max_acc:
                    max_acc=test_acc
                    best_epoch=epoch    
                    find_better=True             
                print '\t\t\ttest_acc:', test_acc, 'max:',    max_acc,'(at',best_epoch,')'
                if find_better==True:
                    store_model_to_file(params, best_epoch, max_acc)
                    print 'Finished storing best params'  

            if patience <= iter:
                done_looping = True
                break
        
        
        print 'Epoch ', epoch, 'uses ', (time.clock()-mid_time)/60.0, 'min'
        mid_time = time.clock()
        #writefile.close()
   
        #print 'Batch_size: ', update_freq
    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


def store_model_to_file(best_params, best_epoch, best_acc):
    save_file = open('/mounts/data/proj/wenpeng/Dataset/MCTest/Best_Para_at'+str(best_epoch)+'_'+str(best_acc), 'wb')  # this will overwrite current contents
    for para in best_params:           
        cPickle.dump(para.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
    save_file.close()

def cosine(vec1, vec2):
    vec1=debug_print(vec1, 'vec1')
    vec2=debug_print(vec2, 'vec2')
    norm_uni_l=T.sqrt((vec1**2).sum())
    norm_uni_r=T.sqrt((vec2**2).sum())
    
    dot=T.dot(vec1,vec2.T)
    
    simi=debug_print(dot/(norm_uni_l*norm_uni_r), 'uni-cosine')
    return simi#.reshape((1,1))    
def Linear(sum_uni_l, sum_uni_r):
    return (T.dot(sum_uni_l,sum_uni_r.T)).reshape((1,1))    
def Poly(sum_uni_l, sum_uni_r):
    dot=T.dot(sum_uni_l,sum_uni_r.T)
    poly=(0.5*dot+1)**3
    return poly.reshape((1,1))    
def Sigmoid(sum_uni_l, sum_uni_r):
    dot=T.dot(sum_uni_l,sum_uni_r.T)
    return T.tanh(1.0*dot+1).reshape((1,1))    
def RBF(sum_uni_l, sum_uni_r):
    eucli=T.sum((sum_uni_l-sum_uni_r)**2)
    return T.exp(-0.5*eucli).reshape((1,1))    
def GESD (sum_uni_l, sum_uni_r):
    eucli=1/(1+T.sum((sum_uni_l-sum_uni_r)**2))
    kernel=1/(1+T.exp(-(T.dot(sum_uni_l,sum_uni_r.T)+1)))
    return (eucli*kernel).reshape((1,1))   
def EUCLID(sum_uni_l, sum_uni_r):
    return T.sqrt(T.sqr(sum_uni_l-sum_uni_r).sum()+1e-20)#.reshape((1,1))
def load_model(params):
    save_file = open('/mounts/data/proj/wenpeng/Dataset/MCTest/Best_Para_at54_0.574583333333')
    
    for para in params:
        para.set_value(cPickle.load(save_file), borrow=True)
    save_file.close() 
def load_model_for_conv2(params):
    #save_file = open('/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/Best_Conv_Para')
    #save_file = open('/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/Best_Conv_Para_at_18')
    save_file = open('/mounts/data/proj/wenpeng/Dataset/SICK/Best_Conv2_Para_20')
    
    for para in params:
        para.set_value(cPickle.load(save_file), borrow=True)
    save_file.close()    

def compute_test_acc(test_y, test_prop):
    size=len(test_y)
    batch=4
    n_batches=size/batch
    
    batch_start=list(numpy.arange(n_batches)*batch)
    corr=0
    for start in batch_start:
        sub_y=test_y[start:start+batch]
        sub_prop=test_prop[start:start+batch]
        big_posi=numpy.argmax(sub_prop)
        if sub_y[big_posi]==1:
            corr+=1
    return corr*1.0/n_batches
        

if __name__ == '__main__':
    evaluate_lenet5()