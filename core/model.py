# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# A is number of attribute features (4 in our case).
# =========================================================================================

from __future__ import division

import tensorflow as tf


class CaptionGenerator(object):
    def __init__(self, word_to_idx, dim_att=[4, 512], dim_feature=[196, 512], dim_embed=512, dim_hidden=1024, n_time_step=16,
                  prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.A = dim_att[0]
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.att_idxs = tf.placeholder(tf.int32, [None, self.A])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])

    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _project_features_ann(self, features):
        with tf.variable_scope('project_features_ann'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            #print '-----------------------------------------------------------'
            #print 'Dimension ann_features_proj: {}'.format(features_proj.get_shape())
            #print '-----------------------------------------------------------'
            return features_proj

    def _attention_layer_ann(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer_ann', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
            #print '-----------------------------------------------------------'
            #print 'Dimension context: {}'.format(context.get_shape())
            #print 'Dimension alpha: {}'.format(alpha.get_shape())
            #print '-----------------------------------------------------------'
            return context, alpha

    def _project_features_att(self, features):
        with tf.variable_scope('project_features_att'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.A, self.D])
            #print '-----------------------------------------------------------'
            #print 'Dimension att_features_proj: {}'.format(features_proj.get_shape())
            #print '-----------------------------------------------------------'
            return features_proj

    def _attention_layer_att(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer_att', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, A, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.A])   # (N, A)
            alpha = tf.nn.softmax(out_att)
            temp = tf.expand_dims(alpha,2)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
            #print '-----------------------------------------------------------'
            #print 'Dimension context: {}'.format(context.get_shape())
            #print 'Dimension alpha: {}'.format(alpha.get_shape())
            #print '-----------------------------------------------------------'
            return context, alpha

    def _build_context(self, context1, context2, h, reuse=False):
        with tf.variable_scope('build_context', reuse=reuse):
            w_ann = tf.get_variable('w_ann', [self.H, 1], initializer=self.weight_initializer)
            b_ann = tf.get_variable('b_ann', [1], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.H, 1], initializer=self.weight_initializer)
            b_att = tf.get_variable('b_att', [1], initializer=self.const_initializer)

            gamma1 = tf.nn.sigmoid(context1 + tf.matmul(h, w_ann) + b_ann, 'gamma1')
            gamma2 = tf.nn.sigmoid(context2 + tf.matmul(h, w_att) + b_att, 'gamma2')

            context = tf.add(tf.mul(gamma1, context1, name='selected_context_1'), \
                             tf.mul(gamma2, context2, name='selected_context_2'))

        return context

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')    # (N, 1)
            context = tf.mul(beta, context, name='selected_context')
            return context, beta

    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))

    def build_model(self):
        ann_feats = self.features
        captions = self.captions
        batch_size = tf.shape(ann_feats)[0]

        captions_in = captions[:, :self.T]
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))

        #Get attribute features
        att_idxs = self.att_idxs
        att_feats = self._word_embedding(inputs=att_idxs)


        # batch normalize feature vectors
        ann_feats = self._batch_norm(ann_feats, mode='train', name='conv_features')
        att_feats = self._batch_norm(att_feats, mode='train', name='emb_features')
        features = tf.concat(1, [ann_feats, att_feats], name='combined_features')
        print 'Joint-features dimension: {}'.format(features.get_shape())

        c, h = self._get_initial_lstm(features=features)
        x = self._word_embedding(inputs=captions_in, reuse=True)
        ann_features_proj = self._project_features_ann(features=ann_feats)
        att_features_proj = self._project_features_att(features=att_feats)

        loss = 0.0
        alpha_list_ann = []
        alpha_list_att = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(self.T):
            context1, alpha1 = self._attention_layer_ann(ann_feats, ann_features_proj, h, reuse=(t!=0))
            context2, alpha2 = self._attention_layer_att(att_feats, att_features_proj, h, reuse=(t!=0))
            context = self._build_context(context1, context2, h, reuse=(t!=0))
            alpha_list_ann.append(alpha1)
            alpha_list_att.append(alpha2)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=tf.concat(1, [x[:,t,:], context]), state=[c, h])

            logits = self._decode_lstm(x[:,t,:], h, context, dropout=self.dropout, reuse=(t!=0))
            loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, captions_out[:, t]) * mask[:, t])

        if self.alpha_c > 0:
            alphas_ann = tf.transpose(tf.pack(alpha_list_ann), (1, 0, 2))     # (N, T, L)
            alphas_all_ann = tf.reduce_sum(alphas_ann, 1)      # (N, L)
            alpha_reg_ann = self.alpha_c * tf.reduce_sum((16./self.L - alphas_all_ann) ** 2)

            alphas_att = tf.transpose(tf.pack(alpha_list_att), (1, 0, 2))     # (N, T, A)
            alphas_all_att = tf.reduce_sum(alphas_att, 1)      # (N, A)
            alpha_reg_att = self.alpha_c * tf.reduce_sum((16./self.A - alphas_all_att) ** 2)

            loss += alpha_reg_ann
            loss += alpha_reg_att

        return loss / tf.to_float(batch_size)

    def build_sampler(self, max_len=20):
        ann_feats = self.features
        att_idxs = self.att_idxs
        att_feats = self._word_embedding(inputs=att_idxs)

        # batch normalize feature vectors
        ann_feats = self._batch_norm(ann_feats, mode='test', name='conv_features')
        att_feats = self._batch_norm(att_feats, mode='test', name='emb_features')
        features = tf.concat(1, [ann_feats, att_feats], name='combined_features')

        c, h = self._get_initial_lstm(features=features)
        ann_features_proj = self._project_features_ann(features=ann_feats)
        att_features_proj = self._project_features_att(features=att_feats)

        sampled_word_list = []
        alpha_list_ann = []
        alpha_list_att = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start), reuse=True)
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            context1, alpha1 = self._attention_layer_ann(ann_feats, ann_features_proj, h, reuse=(t!=0))
            context2, alpha2 = self._attention_layer_att(att_feats, att_features_proj, h, reuse=(t!=0))
            context = self._build_context(context1, context2, h, reuse=(t!=0))
            alpha_list_ann.append(alpha1)
            alpha_list_att.append(alpha2)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))
                beta_list.append(beta)

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=tf.concat(1, [x, context]), state=[c, h])

            logits = self._decode_lstm(x, h, context, reuse=(t!=0))
            sampled_word = tf.argmax(logits, 1)
            sampled_word_list.append(sampled_word)

        alphas_ann = tf.transpose(tf.pack(alpha_list_ann), (1, 0, 2))     # (N, T, L)
        alphas_att = tf.transpose(tf.pack(alpha_list_att), (1, 0, 2))     # (N, T, A)
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        sampled_captions = tf.transpose(tf.pack(sampled_word_list), (1, 0))     # (N, max_len)
        #print alphas_ann.get_shape(), alphas_att.get_shape()
        return alphas_ann, alphas_att, betas, sampled_captions


