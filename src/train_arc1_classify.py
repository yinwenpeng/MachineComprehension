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
from loadData import load_MCTest_corpus, load_word2vec_to_init
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import Conv_with_input_para, Conv_with_input_para_one_col_featuremap, Average_Pooling_for_Top, create_conv_para, Average_Pooling, create_highw_para, Average_Pooling_Scan
from random import shuffle

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy import linalg, mat, dot

#from preprocess_wikiQA import compute_map_mrr

#need to change
'''
1) rouge score
2) TF-KLD

4) paragraph vector

6) tokenized sentences : better
7) only use non-overlap pairs 

9) length of nonoverlap : better




Doesnt work:
3) train+trial
10) no not use mt metrics
5) update word embeddings
8) nonoverlap emb used average
10) delete original sentence lengths

'''

def evaluate_lenet5(learning_rate=0.09, n_epochs=2000, nkerns=[50,50], batch_size=1, window_width=3,
                    maxSentLength=64, maxDocLength=60, emb_size=300, hidden_size=200,
                    margin=0.5, L2_weight=0.00065, update_freq=1, norm_threshold=5.0, max_s_length=57, max_d_length=59):
    maxSentLength=max_s_length+2*(window_width-1)
    maxDocLength=max_d_length+2*(window_width-1)
    model_options = locals().copy()
    print "model options", model_options
    rootPath='/mounts/data/proj/wenpeng/Dataset/MCTest/';
    rng = numpy.random.RandomState(23455)
    train_data,train_size, test_data, test_size, vocab_size=load_MCTest_corpus(rootPath+'vocab.txt', rootPath+'mc500.train.tsv_standardlized.txt', rootPath+'mc500.test.tsv_standardlized.txt', max_s_length,maxSentLength, maxDocLength)#vocab_size contain train, dev and test

    #datasets_nonoverlap, vocab_size_nonoverlap=load_SICK_corpus(rootPath+'vocab_nonoverlap_train_plus_dev.txt', rootPath+'train_plus_dev_removed_overlap_as_training.txt', rootPath+'test_removed_overlap_as_training.txt', max_truncate_nonoverlap,maxSentLength_nonoverlap, entailment=True)
    #datasets, vocab_size=load_wikiQA_corpus(rootPath+'vocab_lower_in_word2vec.txt', rootPath+'WikiQA-train.txt', rootPath+'test_filtered.txt', maxSentLength)#vocab_size contain train, dev and test
    #mtPath='/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/MT/BLEU_NIST/'
#     mt_train, mt_test=load_mts_wikiQA(rootPath+'Train_plus_dev_MT/concate_14mt_train.txt', rootPath+'Test_MT/concate_14mt_test.txt')
#     extra_train, extra_test=load_extra_features(rootPath+'train_plus_dev_rule_features_cosine_eucli_negation_len1_len2_syn_hyper1_hyper2_anto(newsimi0.4).txt', rootPath+'test_rule_features_cosine_eucli_negation_len1_len2_syn_hyper1_hyper2_anto(newsimi0.4).txt')
#     discri_train, discri_test=load_extra_features(rootPath+'train_plus_dev_discri_features_0.3.txt', rootPath+'test_discri_features_0.3.txt')
    #wm_train, wm_test=load_wmf_wikiQA(rootPath+'train_word_matching_scores.txt', rootPath+'test_word_matching_scores.txt')
    #wm_train, wm_test=load_wmf_wikiQA(rootPath+'train_word_matching_scores_normalized.txt', rootPath+'test_word_matching_scores_normalized.txt')
    [train_data_D, train_data_Q, train_data_A, train_Y, train_Label, 
                 train_Length_D,train_Length_D_s, train_Length_Q, train_Length_A,
                train_leftPad_D,train_leftPad_D_s, train_leftPad_Q, train_leftPad_A,
                train_rightPad_D,train_rightPad_D_s, train_rightPad_Q, train_rightPad_A]=train_data
    [test_data_D, test_data_Q, test_data_A, test_Y, test_Label, 
                 test_Length_D,test_Length_D_s, test_Length_Q, test_Length_A,
                test_leftPad_D,test_leftPad_D_s, test_leftPad_Q, test_leftPad_A,
                test_rightPad_D,test_rightPad_D_s, test_rightPad_Q, test_rightPad_A]=test_data                


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
    rand_values=load_word2vec_to_init(rand_values, rootPath+'vocab_embs_300d.txt')
    #rand_values=load_word2vec_to_init(rand_values, rootPath+'vocab_lower_in_word2vec_embs_300d.txt')
    embeddings=theano.shared(value=rand_values, borrow=True)      
    
    #cost_tmp=0
    error_sum=0
    
    # allocate symbolic variables for the data
    index = T.lscalar()
    index_D = T.lmatrix()   # now, x is the index matrix, must be integer
    index_Q = T.lvector()
    index_A= T.lvector()
    y = T.lvector()  
    
    len_D=T.lscalar()
    len_D_s=T.lvector()
    len_Q=T.lscalar()
    len_A=T.lscalar()

    left_D=T.lscalar()
    left_D_s=T.lvector()
    left_Q=T.lscalar()
    left_A=T.lscalar()

    right_D=T.lscalar()
    right_D_s=T.lvector()
    right_Q=T.lscalar()
    right_A=T.lscalar()
        

    #wmf=T.dmatrix()
    cost_tmp=T.dscalar()
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
    layer0_D_input = embeddings[index_D.flatten()].reshape((maxDocLength,maxSentLength, emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)
    layer0_Q_input = embeddings[index_Q.flatten()].reshape((batch_size,maxSentLength, emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)
    layer0_A_input = embeddings[index_A.flatten()].reshape((batch_size,maxSentLength, emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)
    
        
    conv_W, conv_b=create_conv_para(rng, filter_shape=(nkerns[0], 1, filter_words[0], filter_words[1]))
#     load_model_for_conv1([conv_W, conv_b])

    layer0_D = Conv_with_input_para(rng, input=layer0_D_input,
            image_shape=(maxDocLength, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_words[0], filter_words[1]), W=conv_W, b=conv_b)
    layer0_Q = Conv_with_input_para(rng, input=layer0_Q_input,
            image_shape=(batch_size, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_words[0], filter_words[1]), W=conv_W, b=conv_b)
    layer0_A = Conv_with_input_para(rng, input=layer0_A_input,
            image_shape=(batch_size, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_words[0], filter_words[1]), W=conv_W, b=conv_b)
    
    layer0_D_output=debug_print(layer0_D.output, 'layer0_D.output')
    layer0_Q_output=debug_print(layer0_Q.output, 'layer0_Q.output')
    layer0_A_output=debug_print(layer0_A.output, 'layer0_A.output')
    layer0_para=[conv_W, conv_b]    

    layer1_DQ=Average_Pooling_Scan(rng, input_D=layer0_D_output, input_r=layer0_Q_output, kern=nkerns[0],
                                      left_D=left_D, right_D=right_D,
                     left_D_s=left_D_s, right_D_s=right_D_s, left_r=left_Q, right_r=right_Q, 
                      length_D_s=len_D_s+filter_words[1]-1, length_r=len_Q+filter_words[1]-1,
                       dim=maxSentLength+filter_words[1]-1, doc_len=maxDocLength, topk=3)
    layer1_DA=Average_Pooling_Scan(rng, input_D=layer0_D_output, input_r=layer0_A_output, kern=nkerns[0],
                                      left_D=left_D, right_D=right_D,
                     left_D_s=left_D_s, right_D_s=right_D_s, left_r=left_A, right_r=right_A, 
                      length_D_s=len_D_s+filter_words[1]-1, length_r=len_A+filter_words[1]-1,
                       dim=maxSentLength+filter_words[1]-1, doc_len=maxDocLength, topk=3)
    
    conv2_W, conv2_b=create_conv_para(rng, filter_shape=(nkerns[1], 1, nkerns[0], filter_sents[1]))
    #load_model_for_conv2([conv2_W, conv2_b])#this can not be used, as the nkerns[0]!=filter_size[0]
    #conv from sentence to doc
    layer2_DQ = Conv_with_input_para(rng, input=layer1_DQ.output_D.reshape((batch_size, 1, nkerns[0], dshape[1])),
            image_shape=(batch_size, 1, nkerns[0], dshape[1]),
            filter_shape=(nkerns[1], 1, nkerns[0], filter_sents[1]), W=conv2_W, b=conv2_b)
    layer2_DA = Conv_with_input_para(rng, input=layer1_DA.output_D.reshape((batch_size, 1, nkerns[0], dshape[1])),
            image_shape=(batch_size, 1, nkerns[0], dshape[1]),
            filter_shape=(nkerns[1], 1, nkerns[0], filter_sents[1]), W=conv2_W, b=conv2_b)
    #conv single Q and A into doc level with same conv weights
    layer2_Q = Conv_with_input_para_one_col_featuremap(rng, input=layer1_DQ.output_QA_sent_level_rep.reshape((batch_size, 1, nkerns[0], 1)),
            image_shape=(batch_size, 1, nkerns[0], 1),
            filter_shape=(nkerns[1], 1, nkerns[0], filter_sents[1]), W=conv2_W, b=conv2_b)
    layer2_A = Conv_with_input_para_one_col_featuremap(rng, input=layer1_DA.output_QA_sent_level_rep.reshape((batch_size, 1, nkerns[0], 1)),
            image_shape=(batch_size, 1, nkerns[0], 1),
            filter_shape=(nkerns[1], 1, nkerns[0], filter_sents[1]), W=conv2_W, b=conv2_b)
    layer2_Q_output_sent_rep_Dlevel=debug_print(layer2_Q.output_sent_rep_Dlevel, 'layer2_Q.output_sent_rep_Dlevel')
    layer2_A_output_sent_rep_Dlevel=debug_print(layer2_A.output_sent_rep_Dlevel, 'layer2_A.output_sent_rep_Dlevel')
    layer2_para=[conv2_W, conv2_b]
    
    layer3_DQ=Average_Pooling_for_Top(rng, input_l=layer2_DQ.output, input_r=layer2_Q_output_sent_rep_Dlevel, kern=nkerns[1],
                     left_l=left_D, right_l=right_D, left_r=0, right_r=0, 
                      length_l=len_D+filter_sents[1]-1, length_r=1,
                       dim=maxDocLength+filter_sents[1]-1, topk=3)
    layer3_DA=Average_Pooling_for_Top(rng, input_l=layer2_DA.output, input_r=layer2_A_output_sent_rep_Dlevel, kern=nkerns[1],
                     left_l=left_D, right_l=right_D, left_r=0, right_r=0, 
                      length_l=len_D+filter_sents[1]-1, length_r=1,
                       dim=maxDocLength+filter_sents[1]-1, topk=3)
    
    #high-way
    high_W, high_b=create_highw_para(rng, nkerns[0], nkerns[1])
    transform_gate_DQ=debug_print(T.nnet.sigmoid(T.dot(high_W, layer1_DQ.output_D_sent_level_rep) + high_b), 'transform_gate_DQ')
    transform_gate_DA=debug_print(T.nnet.sigmoid(T.dot(high_W, layer1_DA.output_D_sent_level_rep) + high_b), 'transform_gate_DA')
    transform_gate_Q=debug_print(T.nnet.sigmoid(T.dot(high_W, layer1_DQ.output_QA_sent_level_rep) + high_b), 'transform_gate_Q')
    transform_gate_A=debug_print(T.nnet.sigmoid(T.dot(high_W, layer1_DA.output_QA_sent_level_rep) + high_b), 'transform_gate_A')
    highW_para=[high_W, high_b]
        
    overall_D_Q=debug_print((1.0-transform_gate_DQ)*layer1_DQ.output_D_sent_level_rep+transform_gate_DQ*layer3_DQ.output_D_doc_level_rep, 'overall_D_Q')
    overall_D_A=(1.0-transform_gate_DA)*layer1_DA.output_D_sent_level_rep+transform_gate_DA*layer3_DA.output_D_doc_level_rep
    overall_Q=(1.0-transform_gate_Q)*layer1_DQ.output_QA_sent_level_rep+transform_gate_Q*layer2_Q.output_sent_rep_Dlevel
    overall_A=(1.0-transform_gate_A)*layer1_DA.output_QA_sent_level_rep+transform_gate_A*layer2_A.output_sent_rep_Dlevel
    
    simi_sent_level=debug_print(cosine(layer1_DQ.output_D_sent_level_rep+layer1_DA.output_D_sent_level_rep, layer1_DQ.output_QA_sent_level_rep+layer1_DA.output_QA_sent_level_rep), 'simi_sent_level')
    simi_doc_level=debug_print(cosine(layer3_DQ.output_D_doc_level_rep+layer3_DA.output_D_doc_level_rep, layer2_Q.output_sent_rep_Dlevel+layer2_A.output_sent_rep_Dlevel), 'simi_doc_level')
    simi_overall_level=debug_print(cosine(overall_D_Q+overall_D_A, overall_Q+overall_A), 'simi_overall_level')
    

#     eucli_1=1.0/(1.0+EUCLID(layer3_DQ.output_D+layer3_DA.output_D, layer3_DQ.output_QA+layer3_DA.output_QA))
 
    

    
        

    layer4_input=debug_print(T.concatenate([simi_sent_level,
                                simi_doc_level,
                                simi_overall_level
                                ], axis=1), 'layer4_input')#, layer2.output, layer1.output_cosine], axis=1)
    #layer3_input=T.concatenate([mts,eucli, uni_cosine, len_l, len_r, norm_uni_l-(norm_uni_l+norm_uni_r)/2], axis=1)
    #layer3=LogisticRegression(rng, input=layer3_input, n_in=11, n_out=2)
    layer4=LogisticRegression(rng, input=layer4_input, n_in=3, n_out=2)
    
    #L2_reg =(layer3.W** 2).sum()+(layer2.W** 2).sum()+(layer1.W** 2).sum()+(conv_W** 2).sum()
    L2_reg =debug_print((layer4.W** 2).sum()+(high_W**2).sum()+(conv2_W**2).sum()+(conv_W**2).sum(), 'L2_reg')#+(layer1.W** 2).sum()++(embeddings**2).sum()
    cost_this =debug_print(layer4.negative_log_likelihood(y), 'cost_this')#+L2_weight*L2_reg
    cost=debug_print((cost_this+cost_tmp)/update_freq+L2_weight*L2_reg, 'cost')
    #cost=debug_print((cost_this+cost_tmp)/update_freq, 'cost')
    
# 
#     [train_data_D, train_data_Q, train_data_A, train_Y, train_Label, 
#                  train_Length_D,train_Length_D_s, train_Length_Q, train_Length_A,
#                 train_leftPad_D,train_leftPad_D_s, train_leftPad_Q, train_leftPad_A,
#                 train_rightPad_D,train_rightPad_D_s, train_rightPad_Q, train_rightPad_A]=train_data
#     [test_data_D, test_data_Q, test_data_A, test_Y, test_Label, 
#                  test_Length_D,test_Length_D_s, test_Length_Q, test_Length_A,
#                 test_leftPad_D,test_leftPad_D_s, test_leftPad_Q, test_leftPad_A,
#                 test_rightPad_D,test_rightPad_D_s, test_rightPad_Q, test_rightPad_A]=test_data  
#     index = T.lscalar()
#     index_D = T.lmatrix()   # now, x is the index matrix, must be integer
#     index_Q = T.lvector()
#     index_A= T.lvector()
#     
#     y = T.lvector()  
#     len_D=T.lscalar()
#     len_D_s=T.lvector()
#     len_Q=T.lscalar()
#     len_A=T.lscalar()
# 
#     left_D=T.lscalar()
#     left_D_s=T.lvector()
#     left_Q=T.lscalar()
#     left_A=T.lscalar()
# 
#     right_D=T.lscalar()
#     right_D_s=T.lvector()
#     right_Q=T.lscalar()
#     right_A=T.lscalar()
#         
# 
#     #wmf=T.dmatrix()
#     cost_tmp=T.dscalar()
    
    test_model = theano.function([index], [layer4.errors(y),layer4_input, y, layer4.prop_for_posi],
          givens={
            index_D: test_data_D[index], #a matrix
            index_Q: test_data_Q[index],
            index_A: test_data_A[index],
            y: test_Y[index:index+batch_size],
            len_D: test_Length_D[index],
            len_D_s: test_Length_D_s[index],
            len_Q: test_Length_Q[index],
            len_A: test_Length_A[index],

            left_D: test_leftPad_D[index],
            left_D_s: test_leftPad_D_s[index],
            left_Q: test_leftPad_Q[index],
            left_A: test_leftPad_A[index],
        
            right_D: test_rightPad_D[index],
            right_D_s: test_rightPad_D_s[index],
            right_Q: test_rightPad_Q[index],
            right_A: test_rightPad_A[index]
            
            }, on_unused_input='ignore')


    #params = layer3.params + layer2.params + layer1.params+ [conv_W, conv_b]
    params = layer4.params+layer2_para+layer0_para+highW_para
    
    accumulator=[]
    for para_i in params:
        eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
      
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        grad_i=debug_print(grad_i,'grad_i')
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(acc)))   #AdaGrad
        updates.append((acc_i, acc))    
 

  
    train_model = theano.function([index,cost_tmp], cost, updates=updates,
          givens={
            index_D: train_data_D[index],
            index_Q: train_data_Q[index],
            index_A: train_data_A[index],
            y: train_Y[index:index+batch_size],
            len_D: train_Length_D[index],
            len_D_s: train_Length_D_s[index],
            len_Q: train_Length_Q[index],
            len_A: train_Length_A[index],

            left_D: train_leftPad_D[index],
            left_D_s: train_leftPad_D_s[index],
            left_Q: train_leftPad_Q[index],
            left_A: train_leftPad_A[index],
        
            right_D: train_rightPad_D[index],
            right_D_s: train_rightPad_D_s[index],
            right_Q: train_rightPad_Q[index],
            right_A: train_rightPad_A[index]
            }, on_unused_input='ignore')

    train_model_predict = theano.function([index], [cost_this,layer4.errors(y), layer4_input, y],
          givens={
            index_D: train_data_D[index],
            index_Q: train_data_Q[index],
            index_A: train_data_A[index],
            y: train_Y[index:index+batch_size],
            len_D: train_Length_D[index],
            len_D_s: train_Length_D_s[index],
            len_Q: train_Length_Q[index],
            len_A: train_Length_A[index],

            left_D: train_leftPad_D[index],
            left_D_s: train_leftPad_D_s[index],
            left_Q: train_leftPad_Q[index],
            left_A: train_leftPad_A[index],
        
            right_D: train_rightPad_D[index],
            right_D_s: train_rightPad_D_s[index],
            right_Q: train_rightPad_Q[index],
            right_A: train_rightPad_A[index]
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
        #shuffle(train_batch_start)#shuffle training data
        cost_tmp=0.0
#         readfile=open('/mounts/data/proj/wenpeng/Dataset/SICK/train_plus_dev.txt', 'r')
#         train_pairs=[]
#         train_y=[]
#         for line in readfile:
#             tokens=line.strip().split('\t')
#             listt=tokens[0]+'\t'+tokens[1]
#             train_pairs.append(listt)
#             train_y.append(tokens[2])
#         readfile.close()
#         writefile=open('/mounts/data/proj/wenpeng/Dataset/SICK/weights_fine_tune.txt', 'w')
        for batch_start in train_batch_start: 
            # iter means how many batches have been runed, taking into loop
            iter = (epoch - 1) * n_train_batches + minibatch_index +1
            sys.stdout.write( "Training :[%6f] %% complete!\r" % (batch_start*100.0/train_size) )
            sys.stdout.flush()
            minibatch_index=minibatch_index+1
            #if epoch %2 ==0:
            #    batch_start=batch_start+remain_train
            #time.sleep(0.5)
            #print batch_start
            if iter%update_freq != 0:
                cost_ij, error_ij, layer3_input, y=train_model_predict(batch_start)
                #print 'layer3_input', layer3_input
                cost_tmp+=cost_ij
                error_sum+=error_ij

  
            else:
                cost_average= train_model(batch_start,cost_tmp)
                #print 'layer3_input', layer3_input
                error_sum=0
                cost_tmp=0.0#reset for the next batch
                #print 'cost_average ', cost_average
                #print 'cost_this ',cost_this
                #exit(0)
            #exit(0)
            
            if iter % n_train_batches == 0:
                print 'training @ iter = '+str(iter)+' average cost: '+str(cost_average)

            
            if iter % validation_frequency == 0:
                #write_file=open('log.txt', 'w')
                test_losses=[]
                test_y=[]
                test_features=[]
                test_prop=[]
                for i in test_batch_start:
                    test_loss, layer3_input, y, posi_prop=test_model(i)
                    test_prop.append(posi_prop[0][0])
                    #test_losses = [test_model(i) for i in test_batch_start]
                    test_losses.append(test_loss)
                    test_y.append(y[0])
                    test_features.append(layer3_input[0])
                    #write_file.write(str(pred_y[0])+'\n')#+'\t'+str(testY[i].eval())+

                #write_file.close()
                #test_score = numpy.mean(test_losses)
                test_acc=compute_test_acc(test_y, test_prop)
                #test_acc=1-test_score
                print(('\t\t\tepoch %i, minibatch %i/%i, test acc of best '
                           'model %f %%') %
                          (epoch, minibatch_index, n_train_batches,test_acc * 100.))
                #now, see the results of LR
                #write_feature=open(rootPath+'feature_check.txt', 'w')
                 
                train_y=[]
                train_features=[]
                count=0
                for batch_start in train_batch_start: 
                    cost_ij, error_ij, layer3_input, y=train_model_predict(batch_start)
                    train_y.append(y[0])
                    train_features.append(layer3_input[0])
                    #write_feature.write(str(batch_start)+' '+' '.join(map(str,layer3_input[0]))+'\n')
                    #count+=1
 
                #write_feature.close()
                clf = svm.SVC(kernel='linear')#OneVsRestClassifier(LinearSVC()) #linear 76.11%, poly 75.19, sigmoid 66.50, rbf 73.33
                clf.fit(train_features, train_y)
                results=clf.decision_function(test_features)
                lr=linear_model.LogisticRegression(C=1e5)
                lr.fit(train_features, train_y)
                results_lr=lr.decision_function(test_features)
                 
                acc_svm=compute_test_acc(test_y, results)
                acc_lr=compute_test_acc(test_y, results_lr)
 
                find_better=False
                if acc_svm > max_acc:
                    max_acc=acc_svm
                    best_epoch=epoch
                    find_better=True
                if test_acc > max_acc:
                    max_acc=test_acc
                    best_epoch=epoch    
                    find_better=True             
                if acc_lr> max_acc:
                    max_acc=acc_lr
                    best_epoch=epoch
                    find_better=True
                print '\t\t\tsvm:', acc_svm, 'lr:', acc_lr, 'nn:', test_acc, 'max:',    max_acc,'(at',best_epoch,')'
#                 if find_better==True:
#                     store_model_to_file(layer2_para, best_epoch)
#                     print 'Finished storing best conv params'  

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


def store_model_to_file(best_params, best_epoch):
    save_file = open('/mounts/data/proj/wenpeng/Dataset/SICK/Best_Conv2_Para_'+str(best_epoch), 'wb')  # this will overwrite current contents
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
    return simi.reshape((1,1))    
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
    return T.sqrt(T.sqr(sum_uni_l-sum_uni_r).sum()+1e-20).reshape((1,1))
def load_model_for_conv1(params):
    #save_file = open('/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/Best_Conv_Para')
    #save_file = open('/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/Best_Conv_Para_at_18')
    save_file = open('/mounts/data/proj/wenpeng/Dataset/SICK/Best_Conv_Para_13')
    
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