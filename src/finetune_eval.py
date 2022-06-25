import tensorflow as tf
import numpy as np
from finetuner import FineTuneModel, command_line_args, determine_learning_rate, filter_by_layer
from sgd_layerwise import LayerwiseSGD, create_optimizer
import argparse
import re
import time
import os
from modeling import get_activation
import tokenization


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.
  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0, minimum=1e-3):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        self.minimum = minimum
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        print("Epoch %05d: loss: %.5f" % (epoch + 1, current))
        if np.less(current, self.minimum):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print("STOPPING at Epoch %05d: early stopping loss: %.5f" % (self.stopped_epoch + 1, current))
        # else:
        #     self.wait += 1
        #     if self.wait >= self.patience:
        #         self.stopped_epoch = epoch
        #         self.model.stop_training = True
        #         print("Restoring model weights from the end of the best epoch.")
        #         self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


class FineTune(FineTuneModel):

    def __init__(self, config, num_labels):
        super(FineTune, self).__init__(config, use_one_hot_embeddings=False)
        # self.label_embs = tf.compat.v1.get_variable("label_embedding",
        #                              dtype=tf.float32, shape=(num_labels, config.label_emb_size + 1),
        #                              initializer=tf.compat.v1.zeros_initializer())
        self.label_embs = tf.Variable(
            tf.zeros_initializer()(shape=(num_labels, config.label_emb_size + 1), dtype=tf.float32),
            name='label_embs')
        self.trainable_vars = list(self.bert_weights.values()) + list(self.task_weights.values())
        self.trainable_vars = self.filter_by_layer(self.trainable_vars)
        self.trainable_vars.append(self.label_embs)
        # print('Trainable vars:', self.trainable_vars)
    
    def compile(self, optimizer, loss_metric=None, train_metric=None, loss_scale=None,
                test_metric=None, count_metric=None):
        super(FineTune, self).compile(optimizer=optimizer)
        self.loss_metric = loss_metric
        self.train_metric = None
        if train_metric:
            self.train_metric = train_metric
        if loss_scale:
            self.loss_scale = loss_scale
        if test_metric:
            self.test_metric = test_metric
        if count_metric:
            self.count_metric = count_metric
    
    def filter_by_layer(self, weights):
        ''' Return weights to train '''
        weights_dict = {var.name: var for var in weights}
        filtered_weights = filter_by_layer(weights_dict, self.config)
        if self.config.parameter_efficient:
            print('Parameter efficient fine-tuning')
        print('Fine-tuning following weights:')
        print("\n".join(filtered_weights.keys()))
        filtered_weights = list(filtered_weights.values())
        # filtered_weights = []
        # for var in weights:
        #     name = var.name
        #     layer_num = re.findall(r'layer_([0-9]+)', name)
        #     # import ipdb; ipdb.set_trace()
        #     # if not config.warp_layers and config.adapt_layer_norm and len(re.findall('LayerNorm', key)) != 0:
        #     #     filtered_weights[key] = value
        #     if len(layer_num) > 0:
        #         layer_num = int(layer_num[0])
        #         if (len(re.findall('layer_%d/intermediate' % layer_num, name)) > 0 or
        #             len(re.findall('layer_%d/output' % layer_num, name)) > 0):
        #             # do not adapt intermediate MLP
        #             tf.compat.v1.logging.info('Not adapting %s' % name)
        #         else:
        #             # if > min_layer then we train
        #             filtered_weights.append(var)
        #     else:   
        #         filtered_weights.append(var)

        return filtered_weights
    
    @tf.function
    def get_label_embs(self, inputs):
        support_ids = inputs["input_ids"]
        support_mask = inputs["input_mask"]
        support_segment_ids = inputs["segment_ids"]
        support_y = inputs["label_ids"]
        support_num_labels = inputs["num_labels"]
        num_labels = support_num_labels[0]
        support = (support_ids, support_mask, support_segment_ids, support_y)
        _, _, _, label_embs = self.forward(
                support, True, num_labels, weights=self.bert_model.weights,
                task_weights=self.task_weights, reuse=False)
        return label_embs
    
    @tf.function
    def set_label_embs(self, inputs):
        support_ids = inputs["input_ids"]
        support_mask = inputs["input_mask"]
        support_segment_ids = inputs["segment_ids"]
        support_y = inputs["label_ids"]
        support_num_labels = inputs["num_labels"]
        num_labels = support_num_labels[0]
        support = (support_ids, support_mask, support_segment_ids, support_y)
        _, _, _, label_embs = self.forward(
                support, True, num_labels, weights=self.bert_model.weights,
                task_weights=self.task_weights, reuse=False)
        self.label_embs.assign_add(label_embs)
        self.num_labels = num_labels
    
    @tf.function
    def get_logits(self, inputs, training=False):
        support_ids = inputs["input_ids"]
        support_mask = inputs["input_mask"]
        support_segment_ids = inputs["segment_ids"]
        support = (support_ids, support_mask, support_segment_ids, None)
        _, support_logits, _, _ = self.forward(
            support, training, self.num_labels, weights=self.bert_model.weights, 
            label_embs=self.label_embs,
            task_weights=self.task_weights, 
            reuse=True, return_only_logits=True)
        return support_logits
    
    @tf.function
    def call(self, inputs, label_embs=None, create_label_embs=False, training=False):
        # support, query = inputs
        # import ipdb; ipdb.set_trace()
        support_ids = inputs["input_ids"]
        support_mask = inputs["input_mask"]
        support_segment_ids = inputs["segment_ids"]
        support_y = inputs["label_ids"]
        support_num_labels = inputs["num_labels"]
        example_weights = None
        if 'input_weights' in inputs:
            example_weights = inputs['input_weights']

        config = self.config

        num_labels = support_num_labels[0]

        support = (support_ids, support_mask, support_segment_ids, support_y)
        # if create_label_embs:
        #     support_loss, support_logits, support_acc, label_embs, _ = self.forward(
        #         support, training, num_labels, weights=self.bert_model.weights,
        #         task_weights=self.task_weights, reuse=False)
        #     return support_loss, support_acc, label_embs
        # else:
        support_loss, support_logits, support_acc, label_embs = self.forward(
            support, training, num_labels, weights=self.bert_model.weights, 
            label_embs=self.label_embs,
            task_weights=self.task_weights, reuse=True, example_weights=example_weights)
        return support_loss, support_acc, label_embs

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            tf.print('Iteration', self.optimizer.iterations)
            # if tf.equal(self.optimizer.iterations, 0):
            #     _, acc, label_embs = self(data, training=True, create_label_embs=True)
            #     import ipdb; ipdb.set_trace()
            #     assign_op = self.label_embs.assign_add(label_embs)
            #     loss = 0.
            # else:
            loss, acc, label_embs = self(data, label_embs=self.label_embs, training=True)
            raw_loss = loss
            # # Need to do compute average loss as apply_gradients sums per-replica gradients
            # if self.loss_scale:
            #     loss = loss / self.loss_scale
        trainable_vars = self.trainable_vars
        gradients = tape.gradient(loss, trainable_vars)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_metric.update_state(raw_loss)
        if self.train_metric:
            self.train_metric.update_state(acc)

        results = {'loss': loss}
        if self.train_metric:
            results['accuracy'] = self.train_metric.result()
        return results
    
    def test_step(self, data):
        input_ids = data["input_ids"]
        input_mask = data["input_mask"]
        segment_ids = data["segment_ids"]
        y = data["label_ids"]
        # num_labels = data["num_labels"][0]
        self.bert_model.forward(
            weights=self.bert_model.weights,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            reuse=True)
        if not self.config.use_pooled_output:
            all_bert_layers = self.bert_model.get_all_encoder_layers()
            bert_layer_weights = self.task_weights['bert_layer_weights']
            hidden_reps = tf.reduce_sum(
                input_tensor=tf.stack(
                    all_bert_layers, axis=-1)[:, 0, :, :] * bert_layer_weights, axis=-1)
            output_layer = tf.nn.tanh(tf.matmul(hidden_reps, self.task_weights['bert_proj_weight']))
        else:
            output_layer = self.bert_model.get_pooled_output()
        
        output_logits = output_layer
        for wid in range(self.config.output_layers):
            output_logits = tf.matmul(output_logits, self.task_weights['output_weights_%d' % wid], transpose_b=True)
            output_logits = tf.nn.bias_add(output_logits, self.task_weights['output_bias_%d' % wid])
            if wid < self.config.output_layers - 1:
                # output_logits = tf.nn.tanh(output_logits)
                output_logits = get_activation(self.config.activation_fn)(output_logits)
        
        logits = tf.matmul(output_logits,
                           self.label_embs[:, :self.config.label_emb_size],
                           transpose_b=True)
        logits = tf.nn.bias_add(logits, self.label_embs[:, -1])
        y_preds = tf.argmax(logits, -1)
        self.test_metric.update_state(y, y_preds)
        self.count_metric.update_state(tf.shape(y)[0])
        return {'accuracy': self.test_metric.result(), 'count': self.count_metric.result()}

    

def get_dataset(fname, seq_length, num_labels, prefix=None, ori_vocab=None, target_vocab=None, do_lower_case=False):
    name_to_features = {
        # "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([], tf.int64),
    }

    if target_vocab is not None:
        ori_vocab = tokenization.load_vocab(ori_vocab)
        inv_ori_vocab = {v:k for k, v in ori_vocab.items()}
        special_tokens = set(['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[unused0]', '[unused1]'])
        target_vocab = tokenization.load_vocab(target_vocab)
        tokenizer = tokenization.WordpieceTokenizer(vocab=target_vocab)
        
        def map_vocab(*record):
            input_ids, input_mask, segment_ids, num_labels, label_ids = record
            # input_ids = record['input_ids']
            tokens = tokenization.convert_ids_to_tokens(inv_ori_vocab, input_ids)
            if do_lower_case:
                tokens = [x.lower() if x not in special_tokens else x for x in tokens]
            sentence = " ".join(tokens)
            sentence = sentence.replace(" ##", "").strip()
            new_tokens = tokenizer.tokenize(sentence)
            if len(new_tokens) > seq_length:
                if new_tokens[seq_length - 1] == '[PAD]':
                    # there are only pad tokens
                    new_tokens = new_tokens[:seq_length]
                else:
                    # we need to truncate the input and add sep token
                    new_tokens = new_tokens[:seq_length - 1]
                    new_tokens.append('[SEP]')
            elif len(new_tokens) < seq_length:
                while len(new_tokens) < seq_length:
                    new_tokens.append('[PAD]')
            new_input_ids = tokenization.convert_tokens_to_ids(target_vocab, new_tokens)
            if new_tokens[-1] != '[PAD]':
                new_input_mask = [1] * seq_length
            else:
                pad_ind = new_tokens.index('[PAD]')
                new_input_mask = [1] * pad_ind + [0] * (seq_length - pad_ind)
            sep_ind = new_tokens.index('[SEP]')
            new_segment_ids = [0] * (sep_ind + 1) + [1] * (seq_length - sep_ind - 1)
            # record['input_ids'] = new_input_ids
            # record['input_mask'] = new_input_mask
            # record['segment_ids'] = new_segment_ids

            return [new_input_ids, new_input_mask, new_segment_ids, num_labels, label_ids]

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
            if target_vocab is not None:
                output = [out_example['input_ids'], out_example['input_mask'], out_example['segment_ids'], out_example['num_labels'], out_example['label_ids']]
                output = tf.numpy_function(func=map_vocab, 
                                           inp=output, 
                                           Tout=[tf.int64, tf.int64, tf.int64, tf.int32, tf.int64])
                output[0].set_shape([seq_length])
                output[1].set_shape([seq_length])
                output[2].set_shape([seq_length])
                output[3].set_shape([])
                output[4].set_shape([])
                out_example['input_ids'], out_example['input_mask'], out_example['segment_ids'], out_example['num_labels'], out_example['label_ids'] = output
        return out_example

    dataset = tf.data.TFRecordDataset(fname).map(
        lambda x: _decode_record(x, name_to_features, num_labels, prefix=prefix))

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test finetune environment')
    parser = command_line_args(parser)
    parser.add_argument('--support_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--tf2_checkpoint', default='', type=str, 
                        help='tf2 checkpoint directory if want to restore from a tf2 model')
    # parser.add_argument('--num_train_steps', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--k', type=int, required=True, help="k for the k-shot")
    parser.add_argument('--loss_threshold', type=float, default=5e-4, help="threshold to stop fine-tuning, -1 for no threshold")
    parser.add_argument('--target_vocab', required=False, type=str, default='',
                        help="vocab to convert data (only provide if need to convert to another vocab)")
    parser.add_argument('--ori_vocab', required=False, type=str, default='',
                        help="vocab to convert data from (only provide if need to convert to another vocab)")
    parser.add_argument('--do_lower_case', default=False, 
                        type=lambda x: (str(x).lower() == 'true'),
                        help="only provide if need to convert to another vocab")
    config = parser.parse_args()
    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

    support_dataset = get_dataset(config.support_file, config.seq_length,
                                  config.num_labels, 
                                  ori_vocab=config.ori_vocab, target_vocab=config.target_vocab, do_lower_case=config.do_lower_case)
    support_dataset = support_dataset.shuffle(buffer_size=100).batch(
        config.train_batch_size, drop_remainder=True)
    test_dataset = get_dataset(config.test_file, config.seq_length,
                               config.num_labels,
                               ori_vocab=config.ori_vocab, target_vocab=config.target_vocab, do_lower_case=config.do_lower_case)
    test_dataset = test_dataset.batch(config.eval_batch_size)

    model = FineTune(config, config.num_labels)
    if config.tf2_checkpoint:
        print("Reading checkpoint")
        latest_ckpt = tf.train.latest_checkpoint(config.tf2_checkpoint)
        print("Found ckpt:", latest_ckpt)
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(latest_ckpt)
    
    for batch in support_dataset.take(1):
        model.set_label_embs(batch)
    # support_dataset = support_dataset.skip(1)
    model_learning_rates = {}
    for key, value in model.bert_weights.items():
        model_learning_rates[value.name] = determine_learning_rate(
            model.bert_learning_rates, model.task_learning_rates, key).numpy()
    for key, value in model.task_weights.items():
        model_learning_rates[value.name] = determine_learning_rate(
            model.bert_learning_rates, model.task_learning_rates, key).numpy()
    model_learning_rates[model.label_embs.name] = determine_learning_rate(
        model.bert_learning_rates, model.task_learning_rates, 'label_embs').numpy()
    # optimizer = LayerwiseSGD(learning_rate=1.0,
    #                          lr_multipliers=model_learning_rates)
    num_train_steps = int(config.epochs * float(config.k * config.num_labels) / config.train_batch_size)
    print("Total Steps", num_train_steps)
    optimizer = create_optimizer(init_lr=1.0,
                                 num_train_steps=num_train_steps,
                                 num_warmup_steps=int(0.1*num_train_steps),
                                 lr_multipliers=model_learning_rates)
    model.compile(optimizer=optimizer,
                  loss_metric=tf.keras.metrics.Mean(),
                  train_metric=tf.keras.metrics.Mean(),
                  test_metric=tf.keras.metrics.Accuracy(),
                  count_metric=tf.keras.metrics.Sum(dtype=tf.int32))
    if model.optimizer.iterations > 0:
        model.optimizer.set_weights(np.array([0]))
    # epochs = int(config.num_train_steps * float(config.k * config.num_labels) / config.train_batch_size)
    print("Total epochs", config.epochs)
    model.fit(support_dataset, epochs=config.epochs,
              callbacks=[EarlyStoppingAtMinLoss(minimum=config.loss_threshold)],)

    model.evaluate(test_dataset)
    print("test_examples =", model.count_metric.result().numpy())
    print("eval_accuracy =", model.test_metric.result().numpy())
    # import ipdb; ipdb.set_trace()
