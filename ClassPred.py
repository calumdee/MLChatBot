import tensorflow as tf
import numpy as np
from tensorflow.python.layers import core as layers_core

class Model:
    
    source_sequence_length = tf.placeholder(tf.int32, shape=(1))
    decoder_lengths = tf.placeholder(tf.int32, shape=(1))
    encoder_inputs = tf.placeholder(tf.int64, shape=(None))
    target_input = tf.placeholder(tf.int64, shape=(None))
    decoder_outputs = tf.placeholder(tf.int64, shape=(None))
    
    e = tf.reshape(encoder_inputs, [-1,1])
    t = tf.reshape(target_input, [-1,1])

    
    def __init__(self, inVocab, outVocab, nI, nE, lr, sos, eos, max_iter, sess):
        
        j = 0
        self.inList = []
        for word in inVocab:
            if (j!=0) :
                self.inList.append(word.strip())
            j += 1
        self.inVocabSize = j-1
        j = 0
        self.outList = []
        for word in outVocab:
            if (j!=0) :
                self.outList.append(word.strip())
            j += 1
        self.outVocabSize = j-1

        self.inEmbedDim = nE
        self.outEmbedDim = nI

        self.learning_rate=lr
        
        self.embedding_encoder =  np.ndarray(shape=(self.inVocabSize,self.inEmbedDim), dtype=np.float64)

        self.embedding_decoder = np.ndarray(shape=(self.outVocabSize,self.outEmbedDim), dtype=np.float64)
             
        self.tgt_sos_id=sos
        self.tgt_eos_id=eos
        self.maximum_iterations=max_iter
        
        
        # Build RNN cell
        self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.inEmbedDim)

        # Build RNN cell
        self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.outEmbedDim)
    
        # Look up embedding:
        #   encoder_inputs: [max_time, batch_size]
        #   encoder_emb_inp: [max_time, batch_size, embedding_size]
        self.encoder_emb_inp = tf.nn.embedding_lookup(
            self.embedding_encoder, self.e)

        # Look up embedding:
        #   encoder_inputs: [max_time, batch_size]
        #   encoder_emb_inp: [max_time, batch_size, embedding_size]
        self.decoder_emb_inp = tf.nn.embedding_lookup(
            self.embedding_decoder, self.t)
        
        
        # Run Dynamic RNN
        #   encoder_outputs: [max_time, batch_size, num_units]
        #   encoder_state: [batch_size, num_units]
        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
        self.encoder_cell, self.encoder_emb_inp,
        sequence_length=self.source_sequence_length,dtype=np.float64, time_major=True)

        self.projection_layer = layers_core.Dense(
            self.outVocabSize, use_bias=False)
        
        self.sess = sess
        
    def train(self, trainIn, trainOut):
        
        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
            self.decoder_emb_inp, self.decoder_lengths, time_major=True)
        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            self.decoder_cell, helper, self.encoder_state,
            output_layer=self.projection_layer)
        # Dynamic decoding
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
        
        logits = outputs.rnn_output
        
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.decoder_outputs, logits=logits)
        train_loss = (tf.reduce_mean(crossent))
        #print(self.decoder_outputs.shape)
        #print(logits.shape)
        print(train_loss.shape)
        # Calculate and clip gradients
        max_gradient_norm=1

        params = tf.trainable_variables()
        gradients = tf.gradients(train_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, max_gradient_norm)

        # Optimization
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        update_step = optimizer.apply_gradients(
            zip(clipped_gradients, params))
        #update_step = optimizer.minimize(train_loss)
        #update = update_step.outputs
        self.sess.run(tf.global_variables_initializer())
        
        a = 0
        for q in trainIn:
            r = trainOut.readline()
            if (a % 2 == 1) :
                qSplit = q.split()
                rSplit = r.split()
                qIndex = []
                for i in range(len(qSplit)):
                    if (qSplit[i] in self.inList) :
                        qIndex.append(self.inList.index(qSplit[i]))
                    else :
                        qIndex.append(0)
                #qIndex.append(1)        
                rIndex = []
                rIndex.append(1)
                for i in range(len(rSplit)):
                    if (rSplit[i] in self.outList) :
                        rIndex.append(self.outList.index(rSplit[i]))
                    else :
                        rIndex.append(0)
                
                oIndex = []
                oIndex = rIndex[1:]
                oIndex.append(2)

                #rIndex.append(2)
                # no way to check this works correctly without data
                if (a==1): 
                    print(rIndex)
                self.sess.run(update_step, feed_dict={self.encoder_inputs: qIndex, self.target_input: rIndex, self.source_sequence_length: [len(qSplit)], self.decoder_lengths: [len(rSplit)+1], self.decoder_outputs: oIndex } )
             
            a = a + 1
            
    def test(self, testIn):
        
        # Helper
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            self.embedding_decoder,
            tf.fill([1], self.tgt_sos_id), self.tgt_eos_id)

        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            self.decoder_cell, helper, self.encoder_state,
            output_layer=self.projection_layer)
        # Dynamic decoding
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder, maximum_iterations=self.maximum_iterations)
        translations = outputs.sample_id
        translate2 = outputs.rnn_output
        
        chatOut = []
        t = testIn
            
        tSplit = t.split()
        tIndex = []
        for i in range(len(tSplit)):
            if (tSplit[i] in self.inList) :
                 tIndex.append(self.inList.index(tSplit[i]))
            else :
                tIndex.append(0)    
        uIndex = []

        print(tIndex)
        translate = self.sess.run(translations, feed_dict={self.encoder_inputs: tIndex, self.target_input: uIndex, self.source_sequence_length: [len(tSplit)], self.decoder_lengths: [0], self.decoder_outputs: uIndex } )
        print(translate)
        out_sentence = ""
        for i in range(translate.size):
            out_sentence = out_sentence + self.outList[translate[0,i]] + " "
        print(out_sentence)
        chatOut.append(out_sentence)
            
        return chatOut;
    
    def test2(self, testIn):
        
        # Helper
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            self.embedding_decoder,
            tf.fill([1], self.tgt_sos_id), self.tgt_eos_id)

        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            self.decoder_cell, helper, self.encoder_state,
            output_layer=self.projection_layer)
        # Dynamic decoding
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder, maximum_iterations=self.maximum_iterations)
        translations = outputs.sample_id
        translate2 = outputs.rnn_output
        
        chatOut = []
        t = testIn
            
        tSplit = t.split()
        tIndex = []
        for i in range(len(tSplit)):
            if (tSplit[i] in self.inList) :
                tIndex.append(self.inList.index(tSplit[i]))
            else :
                tIndex.append(0)    
        uIndex = []

        print(tIndex)
        translate = self.sess.run(translate2, feed_dict={self.encoder_inputs: tIndex, self.target_input: uIndex, self.source_sequence_length: [len(tSplit)], self.decoder_lengths: [0], self.decoder_outputs: uIndex } )
        print(translate)
                
        catValues = translate[0,0,3:]
        catOutput = np.where(catValues==max(catValues))[0] +3
        print("Output:", catOutput)
        out_sentence = ""
        for i in range(catOutput.size):
            out_sentence = out_sentence + self.outList[catOutput[i]] + " "
        print(out_sentence)
        chatOut.append(out_sentence)
        return chatOut;
