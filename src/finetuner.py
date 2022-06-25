import tensorflow as tf
import modeling
import argparse
from tensorflow.python.saved_model import utils_impl as saved_model_utils
import re
import time
import os

# tf.compat.v1.disable_eager_execution()

def convert_to_tensor(g):
    if isinstance(g, tf.IndexedSlices):
        return tf.convert_to_tensor(value=g)
    else:
        return g


def filter_by_layer(weights, config):
    ''' Return weights to train '''
    filtered_weights = {}
    for key, value in weights.items():
        layer_num = re.findall(r'layer_([0-9]+)', key)
        if config.parameter_efficient or config.adapter_hidden_size > 0:
            # print('Parameter efficient fine-tuning')
            if len(re.findall('LayerNorm', key)) != 0:
                filtered_weights[key] = value
            if config.adapter_hidden_size > 0 and len(re.findall('adapters', key)) != 0 and len(re.findall('global_adapters', key)) == 0:
                filtered_weights[key] = value
        elif not config.warp_layers and config.adapt_layer_norm and len(re.findall('LayerNorm', key)) != 0:
            filtered_weights[key] = value
        elif len(layer_num) > 0:
            layer_num = int(layer_num[0])
            if config.warp_layers and (len(re.findall('layer_%d/intermediate' % layer_num, key)) > 0 or
                                      len(re.findall('layer_%d/output' % layer_num, key)) > 0):
                # do not adapt intermediate MLP
                tf.compat.v1.logging.info('Not adapting %s' % key)
            elif layer_num >= config.min_layer_with_grad:
                # if > min_layer then we train
                filtered_weights[key] = value
        else:
            if len(re.findall('embeddings', key)) == 0 or config.train_word_embeddings:
                # if not embedding layer then train
                filtered_weights[key] = value

    return filtered_weights


def create_task_weights_and_lr(bert_config, noutput, config):
    nhidden = bert_config.hidden_size
    task_weights = {}
    learning_rates = {}
    train_lr = config.train_lr
    for wid in range(config.output_layers):
        outnhidden = nhidden // 2 if wid < config.output_layers - 1 else noutput
        output_weights = tf.compat.v1.get_variable(
            "output_weights_%d" % wid, [outnhidden, nhidden],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.compat.v1.get_variable(
            "output_bias_%d" % wid, [outnhidden], initializer=tf.compat.v1.zeros_initializer())
        task_weights['output_weights_%d' % wid] = output_weights
        task_weights['output_bias_%d' % wid] = output_bias
        learning_rates['output/layer%d' % wid] = []
        for sid in range(config.SGD_K):
            with tf.compat.v1.variable_scope('sgd%d' % sid):
                if config.warp_layers:
                    lr = tf.compat.v1.get_variable("output/layer%d/learning_rate" % wid, dtype=tf.float32, shape=(),
                                         initializer=tf.compat.v1.constant_initializer(0.),
                                         trainable=False)
                else:
                    lr = tf.compat.v1.get_variable("output/layer%d/learning_rate" % wid, dtype=tf.float32, shape=(),
                                         initializer=tf.compat.v1.constant_initializer(train_lr),
                                         trainable=config.is_meta_sgd,
                                         constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if config.clip_lr else None)
                learning_rates['output/layer%d' % wid].append(lr)
                tf.compat.v1.summary.scalar('output_layer%d_sgd%d' % (wid, sid), lr)
        nhidden = outnhidden

    nhidden = bert_config.hidden_size
    for wid in range(config.output_layers):
        outnhidden = nhidden // 2 if wid < config.output_layers - 1 else noutput
        if not config.use_euclidean_norm and wid == config.output_layers - 1:
            tf.compat.v1.logging.info('Not using euclidean norm')
            outnhidden = noutput + 1
        if config.prototypical_baseline:
            tf.compat.v1.logging.info('Using Prototypical Baseline')
            task_weights['label_weights_%d' % wid] = task_weights['output_weights_%d' % wid]
            task_weights['label_bias_%d' % wid] = task_weights['output_bias_%d' % wid]
        else:
            output_weights = tf.compat.v1.get_variable(
                "label_weights_%d" % wid, [outnhidden, nhidden],
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
            output_bias = tf.compat.v1.get_variable(
                "label_bias_%d" % wid, [outnhidden], initializer=tf.compat.v1.zeros_initializer())
            task_weights['label_weights_%d' % wid] = output_weights
            task_weights['label_bias_%d' % wid] = output_bias
        learning_rates['label/layer%d' % wid] = []
        for sid in range(config.SGD_K):
            with tf.compat.v1.variable_scope('sgd%d' % sid):
                if config.prototypical_baseline:
                    learning_rates['label/layer%d' % wid].append(learning_rates['output/layer%d' % wid][sid])
                else:
                    lr = tf.compat.v1.get_variable("label/layer%d/learning_rate" % wid, dtype=tf.float32, shape=(),
                                         initializer=tf.compat.v1.constant_initializer(train_lr),
                                         trainable=config.is_meta_sgd,
                                         constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if config.clip_lr else None)
                    learning_rates['label/layer%d' % wid].append(lr)
                    tf.compat.v1.summary.scalar('label_layer%d_sgd%d' % (wid, sid), lr)
        nhidden = outnhidden

    for wid in range(config.deep_set_layers):
        # outnhidden = nhidden // 2 if wid < config.output_layers - 1 else noutput
        # if not config.use_euclidean_norm and wid == config.output_layers - 1:
        #     tf.logging.info('Not using euclidean norm')
        #     outnhidden = noutput + 1
        output_weights = tf.compat.v1.get_variable(
            "label_set_weights_%d" % wid, [nhidden, nhidden],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.compat.v1.get_variable(
            "label_set_bias_%d" % wid, [nhidden], initializer=tf.compat.v1.zeros_initializer())
        task_weights['label_set_weights_%d' % wid] = output_weights
        task_weights['label_set_bias_%d' % wid] = output_bias
        learning_rates['label/set/layer%d' % wid] = []
        for sid in range(config.SGD_K):
            with tf.compat.v1.variable_scope('sgd%d' % sid):
                lr = tf.compat.v1.get_variable("label/set/layer%d/learning_rate" % wid, dtype=tf.float32, shape=(),
                                     initializer=tf.compat.v1.constant_initializer(train_lr),
                                     trainable=config.is_meta_sgd,
                                     constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if config.clip_lr else None)
                learning_rates['label/set/layer%d' % wid].append(lr)
                tf.compat.v1.summary.scalar('label_set_layer%d_sgd%d' % (wid, sid), lr)

    if not config.use_pooled_output:
        tf.compat.v1.logging.info('Creating Per-layer Weights')
        layer_weights = tf.compat.v1.get_variable(
            "bert_layer_weights", [bert_config.num_hidden_layers],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
        )
        # if not config.do_eval:
        layer_weights = tf.reshape(tf.nn.softmax(layer_weights), [1, 1, -1])
        proj_weight = tf.compat.v1.get_variable(
            "cls_output_projection", [bert_config.hidden_size, bert_config.hidden_size],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
        )
        task_weights['bert_proj_weight'] = proj_weight
        task_weights['bert_layer_weights'] = layer_weights
        learning_rates['bert_layer_weights'] = []
        learning_rates['bert_proj_weight'] = []
        for sid in range(config.SGD_K):
            with tf.compat.v1.variable_scope('sgd%d' % sid):
                lr = tf.compat.v1.get_variable("bert_layer_weights", dtype=tf.float32, shape=(),
                                     initializer=tf.compat.v1.constant_initializer(train_lr),
                                     trainable=config.is_meta_sgd,
                                     constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if config.clip_lr else None)
                learning_rates['bert_layer_weights'].append(lr)
                lr = tf.compat.v1.get_variable("bert_proj_weight", dtype=tf.float32, shape=(),
                                     initializer=tf.compat.v1.constant_initializer(train_lr),
                                     trainable=config.is_meta_sgd,
                                     constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if config.clip_lr else None)
                learning_rates['bert_proj_weight'].append(lr)

    if config.update_only_label_embedding:
        learning_rates['label_embs'] = []
        for sid in range(config.SGD_K):
            with tf.compat.v1.variable_scope('sgd%d' % sid):
                 lr = tf.compat.v1.get_variable("label_embs/learning_rate", dtype=tf.float32, shape=(),
                                      initializer=tf.compat.v1.constant_initializer(train_lr),
                                      trainable=config.is_meta_sgd,
                                      constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if config.clip_lr else None)
                 learning_rates['label_embs'].append(lr)
                 tf.compat.v1.summary.scalar('label_embs_sgd%d' % (sid), lr)

    bert_learning_rates = {}
    for lid in range(bert_config.num_hidden_layers):
        bert_learning_rates[str(lid)] = []
        with tf.compat.v1.variable_scope('layer_%d' % lid):
            for sk in range(config.SGD_K):
                with tf.compat.v1.variable_scope('sgd%d' % sk):
                    lr = tf.compat.v1.get_variable("learning_rate", dtype=tf.float32, shape=(),
                                         initializer=tf.compat.v1.constant_initializer(train_lr),
                                         trainable=config.is_meta_sgd,
                                         constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if config.clip_lr else None)
                    bert_learning_rates[str(lid)].append(lr)
                    tf.compat.v1.summary.scalar('bert_layer%d_sgd%d' % (lid, sk), lr)
    bert_learning_rates["word_embedding"] = []
    bert_learning_rates["pooler"] = []
    for sk in range(config.SGD_K):
        with tf.compat.v1.variable_scope('sgd%d' % sk):
            lr = tf.compat.v1.get_variable("word_embedding_lr", dtype=tf.float32, shape=(),
                                 initializer=tf.compat.v1.constant_initializer(train_lr),
                                 trainable=config.is_meta_sgd,
                                 constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if config.clip_lr else None)
            bert_learning_rates["word_embedding"].append(lr)
            tf.compat.v1.summary.scalar('bert_embeddings_sgd%d' % sk, lr)
            lr = tf.compat.v1.get_variable("pooler_lr", dtype=tf.float32, shape=(),
                                 initializer=tf.compat.v1.constant_initializer(train_lr),
                                 trainable=config.is_meta_sgd,
                                 constraint=lambda x: tf.clip_by_value(x, 1e-8, 1.) if config.clip_lr else None)
            bert_learning_rates["pooler"].append(lr)
            tf.compat.v1.summary.scalar('bert_pooler_sgd%d' % sk, lr)

    return task_weights, learning_rates, bert_learning_rates


def determine_learning_rate(bert_learning_rates, task_learning_rates, key, 
                            sgd_id=0, default_lr=1e-4):
    layer_num = re.findall(r'layer_([0-9]+)', key)
    layer_num = str(int(layer_num[0])) if len(layer_num) != 0 else None
    #import ipdb; ipdb.set_trace();
    if layer_num is not None:
        lr = bert_learning_rates[layer_num][sgd_id]
    elif len(re.findall('embeddings', key)) > 0:
        lr = bert_learning_rates["word_embedding"][sgd_id]
    elif len(re.findall('pooler', key)) > 0:
        lr = bert_learning_rates["pooler"][sgd_id]
    elif len(re.findall('bert_proj_weight', key)) > 0:
        lr = task_learning_rates["bert_proj_weight"][sgd_id]
    elif len(re.findall('bert_layer_weights', key)) > 0:
        lr = task_learning_rates["bert_layer_weights"][sgd_id]
    elif key == 'label_embs':
        lr = task_learning_rates[key][sgd_id]
    else:
        vals = key.split('_')
        if len(vals) >= 3:
            if len(vals) == 3:
                name, _, num = vals
                tname = name + '/layer' + num
            else:
                name1, name2, _, num = vals
                tname = name1 + '/' + name2 + '/layer' + num
            if tname in task_learning_rates:
                lr = task_learning_rates[tname][sgd_id]
            else:
                tf.compat.v1.logging.info('Using default learning rate for %s' % key)
                lr = default_lr
        else:
            tf.compat.v1.logging.info('Using default learning rate for %s' % key)
            lr = default_lr
    return lr


def init_variables_from_ckpt(init_checkpoint):
    tvars = tf.compat.v1.trainable_variables()
    initialized_variable_names = {}
    # init_hook = None
    # scaffold_fn = None
    # if init_checkpoint:
    (assignment_map, initialized_variable_names
        ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    
    tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.compat.v1.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)


class FineTuneModel(tf.keras.Model):

    def __init__(self, config, use_one_hot_embeddings=False):
        super(FineTuneModel, self).__init__()
        self.config = config
        bert_config = modeling.BertConfig.from_json_file(config.bert_config_file)
        task_weights, task_learning_rates, bert_learning_rates = create_task_weights_and_lr(
            bert_config, config.label_emb_size, config)

        bert_model = modeling.BertModel(
            config=bert_config,
            use_one_hot_embeddings=use_one_hot_embeddings,
            adapter_hidden_size=config.adapter_hidden_size,
            global_adapters=config.global_adapters,
            scope="bert")
        self.bert_weights = bert_model.weights
        # import ipdb; ipdb.set_trace()

        self.task_weights = task_weights
        self.task_learning_rates = task_learning_rates
        self.bert_learning_rates = bert_learning_rates
        self.bert_model = bert_model
        if self.config.init_checkpoint:
            self.warm_start()

    def warm_start(self):
        if self.config.init_checkpoint.endswith(".ckpt"):
            checkpoint = self.config.init_checkpoint
            tvars = list(self.bert_model.weights.values())
        else:
            checkpoint = saved_model_utils.get_variables_path(self.config.init_checkpoint)
            tvars = self.weights + list(self.bert_model.weights.values())
        tvars = list(filter(lambda x: 'adapter' not in x.name, tvars))
        if checkpoint is None:
            tf.compat.v1.logging.info('No pre-trained model is available, training from scratch.')
        else:
            tf.compat.v1.logging.info(
                'Pre-trained model found in {0} - warmstarting.'.format(checkpoint))
            tf.compat.v1.train.warm_start(checkpoint, vars_to_warm_start=tvars)

    def get_learning_rate(self, key, sgd_id):
        return determine_learning_rate(self.bert_learning_rates, self.task_learning_rates, 
                                       key, sgd_id)
    
    # def split_inputs(self, inputs):
    #     inputs_a, inputs_b = inputs
    #     support_ids = inputs_a["support_input_ids"]
    #     support_mask = inputs_a["support_input_mask"]
    #     support_segment_ids = inputs_a["support_segment_ids"]
    #     support_y = inputs_a["support_label_ids"]
    #     support_num_labels = inputs_a["support_num_labels"]
    #     support = (support_ids, support_mask, support_segment_ids, support_y, support_num_labels)
    #     print("support_ids:", support_ids.shape)

    #     query_ids = inputs_b["query_input_ids"]
    #     query_mask = inputs_b["query_input_mask"]
    #     query_segment_ids = inputs_b["query_segment_ids"]
    #     query_y = inputs_b["query_label_ids"]
    #     query_num_labels = inputs_b["query_num_labels"]
    #     query = (query_ids, query_mask, query_segment_ids, query_y, query_num_labels)
    #     return support, query

    def forward(self, inputs, is_training, num_labels,
            weights, task_weights=None, reuse=False, label_embs=None, label_mask=None,
            return_per_example_loss=False, create_label_emb=False, return_only_logits=False, 
            example_weights=None):
        _ids_x, _mask_x, _segment_ids_x, _labels = inputs
        self.bert_model.forward(
            weights=weights,
            is_training=is_training,
            input_ids=_ids_x,
            input_mask=_mask_x,
            token_type_ids=_segment_ids_x,
            reuse=reuse)

        if not self.config.use_pooled_output:
            all_bert_layers = self.bert_model.get_all_encoder_layers()
            bert_layer_weights = task_weights['bert_layer_weights']
            # if config.do_eval:
            #     bert_layer_weights = tf.reshape(tf.nn.softmax(bert_layer_weights), [1, 1, -1])
            hidden_reps = tf.reduce_sum(
                input_tensor=tf.stack(
                    all_bert_layers, axis=-1)[:, 0, :, :] * bert_layer_weights, axis=-1)
            # (tasks * batchsize, emb_dim)
            # output_layer = hidden_reps[:, 0, :]
            output_layer = tf.nn.tanh(tf.matmul(hidden_reps, task_weights['bert_proj_weight']))
        else:
            output_layer = self.bert_model.get_pooled_output()
        # hidden_size = output_layer.shape[-1].value
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, rate=1 - self.config.keep_prob)

        output_logits = output_layer
        for wid in range(self.config.output_layers):
            output_logits = tf.matmul(output_logits, task_weights['output_weights_%d' % wid], transpose_b=True)
            output_logits = tf.nn.bias_add(output_logits, task_weights['output_bias_%d' % wid])
            if wid < self.config.output_layers - 1:
                # output_logits = tf.nn.tanh(output_logits)
                output_logits = modeling.get_activation(self.config.activation_fn)(output_logits)

        if label_embs is None or create_label_emb:
            label_logits = output_layer
            for wid in range(self.config.output_layers):
                label_logits = tf.matmul(label_logits, task_weights['label_weights_%d' % wid], transpose_b=True)
                label_logits = tf.nn.bias_add(label_logits, task_weights['label_bias_%d' % wid])
                if wid < self.config.output_layers - 1:
                    # label_logits = tf.nn.tanh(label_logits)
                    label_logits = modeling.get_activation(self.config.activation_fn)(label_logits)

            # unique_labels, mapped_labels, count_labels = tf.unique_with_counts(_labels)
            # label_embs_shape = tf.concat([tf.shape(unique_labels), tf.shape(label_logits)[-1]], axis=-1)
            # label_embs_shape = tf.concat([[num_labels], [tf.shape(label_logits, out_type=tf.int64)[-1]]], axis=-1)
            label_embs_shape = [num_labels, modeling.get_shape_list(label_logits)[-1]]
            _labels_2d = tf.expand_dims(_labels, -1)

            label_embs_new = tf.scatter_nd(indices=_labels_2d, updates=label_logits, shape=label_embs_shape)
            # import ipdb; ipdb.set_trace()
            one_hot_labels = tf.one_hot(_labels, depth=num_labels, dtype=tf.float32)
            # import ipdb; ipdb.set_trace()
            label_counts = tf.transpose(a=tf.reduce_sum(input_tensor=one_hot_labels, axis=0))
            label_embs_new = label_embs_new / tf.maximum(tf.cast(tf.expand_dims(label_counts, -1), tf.float32), 1.0)
            if self.config.deep_set_layers > 0:
                label_embs_init = label_embs_new
                for lid in range(self.config.deep_set_layers):
                    label_embs_new = tf.matmul(label_embs_new, task_weights['label_set_weights_%d' % lid])
                    label_embs_new = tf.nn.bias_add(label_embs_new, task_weights['label_set_bias_%d' % lid])
                    if lid < self.config.deep_set_layers - 1:
                        label_embs_new = modeling.get_activation(self.config.activation_fn)(label_embs_new)
                # add a residual layer
                label_embs_new = label_embs_new + label_embs_init
            if label_embs is None:
                label_embs = label_embs_new
            # if label_mask is None:
            #     label_mask = tf.cast(tf.greater(label_counts, 0), tf.float32)
        else:
            label_embs_new = label_embs

        if return_only_logits:
            _, _edim = label_embs.get_shape().as_list()
            tf.compat.v1.logging.info('Using as softmax parameters, label_emb_dim: %d' % (_edim))
            logits = tf.matmul(output_logits, label_embs[:, :self.config.label_emb_size], transpose_b=True)
            logits = tf.nn.bias_add(logits, label_embs[:, -1])
            return None, logits, None, label_embs_new
        loss, accuracy, logits = self.task_loss(output_logits, label_embs, _labels, None, num_labels, example_weights)
        return loss, logits, accuracy, label_embs_new


    def task_loss(self, features, label_embs, labels, label_mask, num_labels, example_weights):
        with tf.compat.v1.variable_scope("task_loss"):
            if self.config.use_euclidean_norm:
                # import ipdb; ipdb.set_trace()
                logits = -tf.reduce_sum(input_tensor=tf.math.squared_difference(tf.expand_dims(features, -2), tf.expand_dims(label_embs, 0)), axis=-1)
            else:
                # label_emb_norm = tf.norm(label_embs, axis=-1, keep_dims=True) + 1e-6
                # label_embs = label_embs / label_emb_norm
                _, _edim = label_embs.get_shape().as_list()
                tf.compat.v1.logging.info('Using as softmax parameters, label_emb_dim: %d' % (_edim))
                logits = tf.matmul(features, label_embs[:, :self.config.label_emb_size], transpose_b=True)
                logits = tf.nn.bias_add(logits, label_embs[:, -1])
            if self.config.stop_grad:
                per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            else:
                # probabilities = tf.nn.softmax(logits, axis=-1)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
                per_example_loss = -tf.reduce_sum(input_tensor=one_hot_labels * log_probs, axis=-1)

            # example_mask = tf.scatter_nd(indices=labels, updates=label_mask, shape=tf.shape(labels))
            # example_mask = tf.gather(indices=labels, params=label_mask)
            per_example_loss = per_example_loss  # * example_mask
            if example_weights is not None:
                per_example_loss *= example_weights
            # accuracy = tf.contrib.metrics.accuracy(tf.argmax(input=logits, axis=1, output_type=tf.int64), labels)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(input=logits, axis=1, output_type=tf.int64), labels), tf.float32))
            if self.config.debug:
                tctx = tf.distribute.get_replica_context()
                tower_id = tf.get_static_value(tctx.replica_id_in_sync_group)
                print_op = tf.print("Tower: %d, task: %d" % (tower_id, 0), {#'per_example_loss': per_example_loss,
                                                                })
                with tf.control_dependencies([print_op]):
                    loss = tf.reduce_mean(input_tensor=per_example_loss)
            else:
                loss = tf.reduce_mean(input_tensor=per_example_loss)
            # loss = tf.reduce_sum(per_example_loss * example_mask) / tf.maximum(tf.reduce_sum(example_mask), 1.0)
        return loss, accuracy, logits

    # @tf.function
    # def call(self, input_ids, input_mask, segement_ids, num_labels, label_ids, training=False):
    @tf.function
    def call(self, inputs, training=False):
        # support, query = inputs
        input_ids, input_mask, segement_ids, num_labels, label_ids = inputs
        support, query = self.split_inputs(input_ids, input_mask, segement_ids, num_labels, label_ids)
        print(support)
        print(query)
        # import ipdb; ipdb.set_trace()
        support_ids, support_mask, support_segment_ids, support_y, support_num_labels = support
        query_ids, query_mask, query_segment_ids, query_y, query_num_labels = query

        config = self.config

        # if config.max_train_batch_size > 0:
            # idx = tf.random.shuffle(tf.range(config.max_train_batch_size))
        # else:
        idx = tf.random.shuffle(tf.range(support_ids.shape[0]))
        support_ids = tf.gather(support_ids, idx, axis=0)
        support_mask = tf.gather(support_mask, idx, axis=0)
        support_segment_ids = tf.gather(support_segment_ids, idx, axis=0)
        support_y = tf.gather(support_y, idx, axis=0)

        num_labels = query_num_labels[0]

        support_ids = tf.reshape(support_ids,
                                 [config.num_batches,
                                  support_ids.get_shape().as_list()[0] // config.num_batches] +
                                 support_ids.get_shape().as_list()[1:])
        support_mask = tf.reshape(support_mask,
                                  [config.num_batches,
                                   support_mask.get_shape().as_list()[0] // config.num_batches] +
                                  support_mask.get_shape().as_list()[1:])
        support_segment_ids = tf.reshape(support_segment_ids,
                                         [config.num_batches,
                                          support_segment_ids.get_shape().as_list()[0] // 
                                          config.num_batches] +
                                         support_segment_ids.get_shape().as_list()[1:])
        support_y = tf.reshape(support_y,
                               [config.num_batches,
                                support_y.get_shape().as_list()[0] // config.num_batches] +
                               support_y.get_shape().as_list()[1:])
        support = (support_ids[0], support_mask[0], support_segment_ids[0], support_y[0])
        query = (query_ids, query_mask, query_segment_ids, query_y)
        # with tf.GradientTape as finetune_tape:
        support_loss, support_logits, support_acc, label_embs = self.forward(
            support, training, num_labels, weights=self.bert_model.weights,
            task_weights=self.task_weights, reuse=False)
        if config.update_only_label_embedding:
            current_task_weights = {key: value for key, value in self.task_weights.items() 
                                    if "label" not in key}
        else:
            current_task_weights = {key: value for key, value in self.task_weights.items()}
        weights_for_grad = filter_by_layer(self.bert_model.weights, config)
        tf.compat.v1.logging.info("Fine-tuning these vars:\n" + "\n".join(weights_for_grad.keys()))
        # import ipdb; ipdb.set_trace()

        if config.sgd_first_batch:
            # import ipdb; ipdb.set_trace()
            grads = tf.gradients(ys=support_loss, xs=list(weights_for_grad.values()))

            task_grads = tf.gradients(ys=support_loss, xs=list(current_task_weights.values()))
            all_grads = grads + task_grads
            if config.update_only_label_embedding:
                label_emb_grads = tf.gradients(ys=support_loss, xs=label_embs)
                # label_emb_grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in label_emb_grads]
                all_grads += label_emb_grads

            if config.stop_grad:
                new_grads = []
                for g in all_grads:
                    if g is None:
                        new_grads.append(g)
                    elif isinstance(g, tf.IndexedSlices):
                        new_grads.append(
                            tf.IndexedSlices(tf.stop_gradient(g.values), g.indices, dense_shape=g.dense_shape))
                    else:
                        new_grads.append(tf.stop_gradient(g))
                all_grads = new_grads
            else:
                tf.compat.v1.logging.info('Using second order')

            all_grads, _ = tf.clip_by_global_norm(all_grads, clip_norm=1.0)

            grads = all_grads[:len(grads)]
            task_grads = all_grads[len(grads):len(grads) + len(task_grads)]
            label_emb_grads = None
            if config.update_only_label_embedding:
                label_emb_grads = all_grads[len(grads) + len(task_grads):]

            # grad and variable dict
            gvs = dict(zip(weights_for_grad.keys(), grads))
            task_gvs = dict(zip(current_task_weights.keys(), task_grads))

            # theta = theta - alpha * grads
            # import ipdb; ipdb.set_trace()
            fast_weights = dict(zip(weights_for_grad.keys(),
                                    [weights_for_grad[key] - 
                                     self.get_learning_rate(key, 0) * convert_to_tensor(gvs[key])
                                     if gvs[key] is not None else weights_for_grad[key]
                                     for key in weights_for_grad.keys()]))

            fast_task_weights = {key:
                                 param - self.get_learning_rate(key, 0) * convert_to_tensor(task_gvs[key])
                                 for key, param in current_task_weights.items()}
            if config.update_only_label_embedding:
                label_embs = label_embs - \
                                self.get_learning_rate('label_embs', 0) * label_emb_grads[0]
        else:
            # SGD_K == 0 : no adaptation
            fast_weights = weights_for_grad
            fast_task_weights = current_task_weights
        fast_weights_forward = {key: fast_weights[key] 
                                if key in fast_weights else self.bert_model.weights[key]
                                for key in self.bert_model.weights}

        query_loss = tf.constant(0., dtype=tf.float32)
        nquery_steps = 0
        query_weight_sum = 0.
        total_inner_steps = config.inner_epochs * (config.num_batches - 1) + 1
        tf.compat.v1.logging.info('total_inner_steps: %d' % total_inner_steps)
        if config.stop_grad and (config.average_query_loss or config.warp_layers or config.prototypical_baseline):
            step_query_loss, query_logits, query_acc, _ = self.forward(query,
                                                                     training, num_labels,
                                                                     weights=fast_weights_forward,
                                                                     task_weights=fast_task_weights,
                                                                     label_embs=label_embs,
                                                                    #  label_mask=label_mask,
                                                                     reuse=True)
            # tf.summary.scalar('step%d_query_loss' % 0, step_query_loss)
            tf.compat.v1.logging.info('Query loss at step %d' % 0)
            if config.weight_query_loss:
                q_wt = (1. / 2**total_inner_steps)
                step_query_loss = q_wt * step_query_loss
                query_weight_sum += q_wt
                # tf.summary.scalar('step%d_query_loss_weight' % 0, q_wt)
                tf.compat.v1.logging.info('\t query loss weight: %.5f' % q_wt)
            query_loss += step_query_loss
            nquery_steps += 1

        lr_multiplier = tf.constant(1.0, dtype=tf.float32)
        num_inner_steps = config.num_batches - 1
        if config.randomize_inner_steps and training:
            num_inner_steps = tf.random.uniform(shape=(), minval=config.min_inner_steps,
                                                maxval=config.num_batches, dtype=tf.int32)

        # continue to build G steps graph
        for ep_id in range(config.inner_epochs):
            if config.num_batches > 0 and ep_id > 0:
                # shuffle batches before each epoch
                _idx = tf.random.shuffle(tf.range(config.num_batches))
                # import ipdb; ipdb.set_trace()
                support_ids = tf.gather(support_ids, _idx, axis=0)
                support_mask = tf.gather(support_mask, _idx, axis=0)
                support_segment_ids = tf.gather(support_segment_ids, _idx, axis=0)
                support_y = tf.gather(support_y, _idx, axis=0)
                # num_labels = tf.gather(num_labels, idx, axis=1)
            for batch_id in range(1, config.num_batches if config.num_batches > 0 else config.SGD_K):
                # we need meta-train loss to fine-tune the task and meta-test loss to update theta
                if config.num_batches > 0:
                    support = (support_ids[batch_id], support_mask[batch_id], support_segment_ids[batch_id],
                               support_y[batch_id])
                if config.update_only_label_embedding:
                    # TODO: CORRECT LABEL MASK TO INCLUDE LABEL NOT SEEN DURING FIRST FORWARD PASS
                    support_loss, support_logits, support_acc, _ = self.forward(
                        support, training, num_labels, 
                        weights=fast_weights_forward, label_embs=label_embs, #label_mask=label_mask, 
                        task_weights=fast_task_weights, reuse=True)
                else:
                    support_loss, support_logits, support_acc, label_embs = self.forward(
                        support, training, num_labels, 
                        weights=fast_weights_forward, task_weights=fast_task_weights, reuse=True)
                # compute gradients
                grads = tf.gradients(ys=support_loss, xs=list(fast_weights.values()))
                task_grads = tf.gradients(ys=support_loss, xs=list(fast_task_weights.values()))
                all_grads = grads + task_grads
                if config.update_only_label_embedding:
                    label_emb_grads = tf.gradients(ys=support_loss, xs=label_embs)
                    # label_emb_grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in label_emb_grads]
                    all_grads += label_emb_grads

                if config.stop_grad:
                    new_grads = []
                    for g in all_grads:
                        if g is None:
                            new_grads.append(g)
                        elif isinstance(g, tf.IndexedSlices):
                            new_grads.append(
                                tf.IndexedSlices(tf.stop_gradient(g.values), g.indices, dense_shape=g.dense_shape))
                        else:
                            new_grads.append(tf.stop_gradient(g))
                    all_grads = new_grads
                else:
                    tf.compat.v1.logging.info('Using second order')

                all_grads, _ = tf.clip_by_global_norm(all_grads, clip_norm=1.0)

                grads = all_grads[:len(grads)]
                task_grads = all_grads[len(grads):len(grads) + len(task_grads)]
                label_emb_grads = None
                if config.update_only_label_embedding:
                    label_emb_grads = all_grads[len(grads) + len(task_grads):]

                # compose grad and variable dict
                gvs = dict(zip(fast_weights.keys(), grads))
                task_gvs = dict(zip(fast_task_weights.keys(), task_grads))
                # import ipdb; ipdb.set_trace()
                # update theta_pi according to varibles
                if config.randomize_inner_steps and training:
                    lr_multiplier = tf.cond(pred=num_inner_steps < batch_id, true_fn=lambda: tf.constant(0.), false_fn=lambda: tf.constant(1.0))
                # print_op = tf.print("Inner step %d" % batch_id, {'lr_multiplier': lr_multiplier,
                #                                                  'num_inner_steps': num_inner_steps})
                lr_id = batch_id if config.num_batches == config.SGD_K else 0
                # with tf.control_dependencies([print_op]):
                fast_weights = dict(zip(fast_weights.keys(),
                                        [fast_weights[key] - lr_multiplier * self.get_learning_rate(key, lr_id) *
                                         convert_to_tensor(gvs[key]) if gvs[key] is not None else fast_weights[key]
                                         for key in fast_weights.keys()]))
                # import ipdb; ipdb.set_trace()
                if config.update_only_label_embedding:
                    not_labels = ["label" not in key for key in fast_task_weights.keys()]
                    assert all(not_labels), "label key exists!"

                fast_task_weights = {key: param - lr_multiplier * self.get_learning_rate(key, lr_id) *
                                     convert_to_tensor(task_gvs[key])
                                     for key, param in fast_task_weights.items()}
                # forward on theta_pi
                if config.update_only_label_embedding:
                    label_embs = label_embs - \
                                 lr_multiplier * self.get_learning_rate('label_embs', lr_id) * label_emb_grads[0]
                fast_weights_forward = {key: fast_weights[key] 
                                        if key in fast_weights else self.bert_model.weights[key]
                                        for key in self.bert_model.weights}
                # we need accumulate all meta-test losses to update theta
                if (ep_id == config.inner_epochs - 1 and batch_id == config.num_batches - 1) or \
                        ((config.average_query_loss or config.warp_layers) and
                         (config.num_batches * ep_id + batch_id) % config.average_query_every == 0):
                    step_query_loss, query_logits, query_acc, _ = self.forward(
                        query, training, num_labels,
                        weights=fast_weights_forward, task_weights=fast_task_weights,
                        label_embs=label_embs, 
                        # label_mask=label_mask, 
                        reuse=True)
                    # tf.summary.scalar('step%d_query_loss' % (ep_id * (config.num_batches - 1) + batch_id),
                    #                   step_query_loss)
                    # tf.compat.v1.logging.info('Query loss at step ep_id %d, batch_id %d' % (ep_id, batch_id))
                    if config.weight_query_loss:
                        q_wt = 1. / 2 ** float(total_inner_steps - ep_id * (config.num_batches - 1) - batch_id)
                        step_query_loss = q_wt * step_query_loss
                        query_weight_sum += q_wt
                        # tf.summary.scalar('step%d_query_loss_weight' % (ep_id * (config.num_batches - 1) + batch_id),
                        #                   q_wt)
                        # tf.compat.v1.logging.info('\t query loss weight: %.5f' % q_wt)
                    query_loss += step_query_loss
                    nquery_steps += 1
                # query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_y)
                # query_preds.append(query_pred)
                # query_losses.append(query_loss)

        if config.average_query_loss or config.warp_layers:
            if config.weight_query_loss:
                query_loss /= query_weight_sum
            else:
                query_loss /= nquery_steps

        query_pred = tf.argmax(input=query_logits, axis=1, output_type=tf.int64)

        return query_loss, query_acc, query_pred
    
    # @tf.function
    # def train_step(self, data):
    #     with tf.GradientTape() as tape:
    #         loss, query_acc, query_pred = self(data, training=True)
        
    #     trainable_vars = self.trainable_variables + list(self.bert_model.weights.values())
    #     print(trainable_vars)
    #     gradients = tape.gradient(loss, trainable_vars)
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    #     # import ipdb; ipdb.set_trace()
    #     # self.compiled_metrics.update_state(loss, loss)

    #     # results = {m.name: m.result() for m in self.metrics}
    #     # results['query_acc'] = query_acc
    #     return loss


class MAMLFineTuner(FineTuneModel):
    
    def compile(self, optimizer, loss_metric=None, train_metric=None, loss_scale=None):
        super(MAMLFineTuner, self).compile(optimizer=optimizer)
        self.loss_metric = loss_metric
        self.train_metric = train_metric
        if loss_scale:
            self.loss_scale = loss_scale
    
    def split_by_meta_batchsz(self, x):
        # if x.get_shape().ndims > 1:
        if self.config.max_train_batch_size > 0:
            x = x[:self.config.max_train_batch_size]
        support, query = x[: self.config.meta_batchsz], x[self.config.meta_batchsz:]
        return support, query

    def split_inputs(self, input_ids, input_mask, input_segment_ids, num_labels, label_ids):
        # print(inputs)
        # input_ids = inputs['input_ids']
        # input_mask = inputs['input_mask']
        # input_segment_ids = inputs['segment_ids']
        # label_ids = inputs['label_ids']
        # num_labels = inputs['num_labels']
        # input_ids, input_mask, input_segment_ids, num_labels, label_ids = inputs
        support_ids, query_ids = self.split_by_meta_batchsz(input_ids)
        support_mask, query_mask = self.split_by_meta_batchsz(input_mask)
        support_segment_ids, query_segment_ids = self.split_by_meta_batchsz(input_segment_ids)
        support_y, query_y = self.split_by_meta_batchsz(label_ids)
        support_num_labels, query_num_labels = self.split_by_meta_batchsz(num_labels)
        support = (support_ids, support_mask, support_segment_ids, support_y, support_num_labels)
        query = (query_ids, query_mask, query_segment_ids, query_y, query_num_labels)
        # import ipdb; ipdb.set_trace()
        return support, query

    # @tf.function
    # def train_step(self, input_ids, input_mask, segement_ids, num_labels, label_ids):
    # @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            # loss, query_acc, query_pred = self(input_ids, input_mask, segement_ids, num_labels, label_ids, training=True)
            loss, query_acc, query_pred = self(inputs, training=True)
            raw_loss = loss
            # Need to do compute average loss as apply_gradients sums per-replica gradients
            if self.loss_scale:
                loss = loss / self.loss_scale
        
        trainable_vars = self.trainable_variables # + list(self.bert_model.weights.values())
        # print(trainable_vars)
        gradients = tape.gradient(loss, trainable_vars)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_metric.update_state(raw_loss)
        self.train_metric.update_state(query_acc)

        # results = {m.name: m.result() for m in self.metrics}
        # results['query_acc'] = query_acc
        return loss
        # return results


def sample_data_generator(fname, seq_length, num_labels, prefix=None):
    name_to_features = {
        # "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features, nlabels, prefix=None):
        '''Decodes a record to a TensorFlow example.'''
        example = tf.io.parse_single_example(serialized=record, features=name_to_features)
        example['num_labels'] = nlabels
        out_example = {}
        if prefix:
            for key, value in example.items():
                out_example[prefix + key] = value
        else:
            out_example = example
        return out_example

    dataset = tf.data.TFRecordDataset(fname).map(
        lambda x: _decode_record(x, name_to_features, num_labels, prefix=prefix))

    # query_dataset = tf.data.TFRecordDataset(query_fname).map(
    #     lambda x: _decode_record(x, name_to_features, num_labels, prefix="query_"))

    # support_dataset = support_dataset.shuffle(buffer_size=100).batch(config.train_batch_size)
    # query_dataset = query_dataset.shuffle(buffer_size=100).batch(config.eval_batch_size)

    # dataset = tf.data.Dataset.zip((support_dataset, query_dataset))

    return dataset


def command_line_args(parser):
    parser.add_argument('--bert_config_file', required=True, type=str)
    parser.add_argument('--init_checkpoint', default="", required=False, type=str)
    parser.add_argument('--seq_length', default=128, type=int)
    parser.add_argument('--train_batch_size', default=90, type=int, help="size of support data")
    parser.add_argument('--eval_batch_size', default=32, type=int, help="size of query data")
    parser.add_argument('--max_train_batch_size', default=-1, type=int, help="size of support data")
    parser.add_argument('--num_batches', default=9, type=int, help='num steps of fine-tuning')
    parser.add_argument('--label_emb_size', default=256, type=int, help='size of label emb')
    parser.add_argument('--tasks_per_gpu', default=1, type=int)
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--warp_layers', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--average_query_loss', default=False, 
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--weight_query_loss', default=True, 
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--average_query_every', default=1, type=int)
    parser.add_argument('--inner_epochs', default=1, type=int)
    parser.add_argument('--update_only_label_embedding', default=True, 
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--stop_grad', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--sgd_first_batch', default=False, 
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--SGD_K', default=1, type=int)
    parser.add_argument('--randomize_inner_steps', default=False, 
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--num_labels', default=2, type=int)
    parser.add_argument('--train_lr', default=1e-5, type=float)
    parser.add_argument('--keep_prob', default=0.9, type=float)
    parser.add_argument('--use_pooled_output', default=True, 
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--train_word_embeddings', default=True, 
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--debug', default=False, 
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--min_layer_with_grad', default=0, type=int)
    parser.add_argument('--adapt_layer_norm', default=False, 
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--output_layers', default=2, type=int)
    parser.add_argument('--use_euclidean_norm', default=False, 
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--is_meta_sgd', default=True, 
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--prototypical_baseline', default=False, 
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--deep_set_layers', default=0, type=int)
    parser.add_argument('--activation_fn', default="tanh", type=str)
    parser.add_argument('--clip_lr', default=True, 
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--parameter_efficient', default=False, 
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--adapter_hidden_size', default=-1, type=int,
                        help='hidden size for adapters, pass -1 to not use adapters')
    parser.add_argument('--global_adapters', default=False, 
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--learn_adapter_init', default=False, 
                        type=lambda x: (str(x).lower() == 'true'))
                        

    return parser



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test finetune environment')
    parser = command_line_args(parser)
    parser.add_argument('--support_file', type=str, required=True)
    parser.add_argument('--query_file', type=str, required=True)
    parser.add_argument('--tf2_checkpoint', default='', type=str, 
                        help='tf2 checkpoint directory if want to restore from a tf2 model')
    config = parser.parse_args()
    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

    support_dataset = None
    if not os.path.isdir(config.support_file):
        support_dataset = sample_data_generator(
            config.support_file, config.seq_length, 
            config.num_labels, prefix="support_")
        support_dataset = support_dataset.shuffle(buffer_size=100).batch(config.train_batch_size, drop_remainder=True).repeat(100)
    else:
        fnames = [os.path.join(config.support_file, x) 
                  for x in os.listdir(config.support_file) if x.endswith('.tfrecord')]
        for i, fname in enumerate(fnames):
            s_dataset = sample_data_generator(
                fname, config.seq_length, 
                config.num_labels, prefix="support_")
            s_dataset = s_dataset.shuffle(buffer_size=100).take(60)
            # support_datasets.append(s_dataset)
            if i ==0:
                support_dataset = s_dataset
            else:
                support_dataset = support_dataset.concatenate(s_dataset)
        # import ipdb; ipdb.set_trace()
        support_dataset = support_dataset.shuffle(buffer_size=1000)
        support_dataset = support_dataset.batch(config.train_batch_size, drop_remainder=True).shuffle(buffer_size=100).repeat()

    query_dataset = sample_data_generator(
        config.query_file, config.seq_length, 
        config.num_labels, prefix="query_")
    query_dataset = query_dataset.shuffle(buffer_size=100).batch(config.eval_batch_size)

    dataset = tf.data.Dataset.zip((support_dataset, query_dataset))

    model = FineTuneModel(config)
    if config.tf2_checkpoint:
        print("Reading checkpoint")
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(tf.train.latest_checkpoint(config.tf2_checkpoint)).assert_existing_objects_matched()
    sum_accuracy = tf.keras.metrics.Sum()
    total_query_examples = 0
    num_ones = 0

    for ex in dataset:
        # import ipdb; ipdb.set_trace()
        start_time = time.time()
        result = model(ex)
        ctime = time.time() - start_time
        sum_accuracy.update_state(result[1] * ex[1]['query_input_ids'].shape[0])
        total_query_examples += ex[1]['query_input_ids'].shape[0]
        num_ones += (ex[1]['query_label_ids'] == 1).numpy().sum()
        accuracy = sum_accuracy.result().numpy() / total_query_examples
        print("Loss: {0}, Accuracy: {1}, Time: {2}s".format(result[0].numpy(), 
                                                            accuracy, ctime))
    accuracy = sum_accuracy.result().numpy() / total_query_examples
    print("Ones accuracy:", num_ones * 1.0 / total_query_examples)
    print("Total Examples: {0}, Accuracy: {1}".format(total_query_examples, accuracy))
