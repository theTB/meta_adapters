import numpy as np
import tensorflow as tf
from tensorflow.python.ops.ragged import segment_id_ops
from finetuner import MAMLFineTuner, command_line_args
import argparse
import os
import sys
import json
import time
import optimization
import tokenization
import horovod.tensorflow as hvd
#import wandb
from collections import OrderedDict

# tf.compat.v1.disable_eager_execution()
hvd.init()

def get_dataset(data_files, sampling_weights, pretrain_task_size, seq_length, ori_vocab=None, target_vocab=None, do_lower_case=False):
    name_to_features = {
        # "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([], tf.int64),
    }

    _label_map = {'mnli': 3, 'sst': 2, 'mrpc': 2, 'winograd': 2, 'rte': 2, 'snli': 3, 'quora': 2,
                'acceptability': 2, 'sts': 5, 'amazon': 2, 'squad': 2,
                'asv': 3, 'hst': 3, 'svt': 3, 'ahs': 3, 'aht': 3, 'hsv': 3,
                'fewrel': 5
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

    all_datasets = []
    sampled_dataset = None
    for full_fname in data_files:
        fname = os.path.splitext(os.path.basename(full_fname))[0]
        nlabels = int(fname.split("_")[-1])
        
        def _decode_record(record, prefix=None):
            '''Decodes a record to a TensorFlow example.'''
            example = tf.io.parse_single_example(serialized=record, features=name_to_features)
            example['num_labels'] = nlabels
            out_example = {}
            if prefix:
                for key, value in example.items():
                    out_example[prefix + key] = value
            else:
                out_example = example
            output = [out_example['input_ids'], out_example['input_mask'], out_example['segment_ids'], out_example['num_labels'], out_example['label_ids']]
            if target_vocab is not None:
                output = tf.numpy_function(func=map_vocab, 
                                           inp=output, 
                                           Tout=[tf.int64, tf.int64, tf.int64, tf.int32, tf.int64])
                output[0].set_shape([seq_length])
                output[1].set_shape([seq_length])
                output[2].set_shape([seq_length])
                output[3].set_shape([])
                output[4].set_shape([])
            return output
        
        dataset = tf.data.TFRecordDataset(full_fname).map(_decode_record, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True)
        if not fname.startswith('meta'):
            print("SHUFFLING:", fname)
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(pretrain_task_size, drop_remainder=True).repeat()
        # TODO: add a snapshot here
        # if sampled_dataset is None:
        #     sampled_dataset = dataset
        # else:
        #     sampled_dataset = sampled_dataset.concatenate(dataset)
        all_datasets.append(dataset)
    
    sampled_dataset = tf.data.experimental.sample_from_datasets(all_datasets, sampling_weights)

    return sampled_dataset


def _is_chief(task_type, task_id):
    # If `task_type` is None, this may be operating as single worker, which works 
    # effectively as chief.
    return task_type is None or task_type == 'chief' or (
        task_type == 'worker' and task_id == 0)


def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir


def write_filepath(filepath, task_type, task_id):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_chief(task_type, task_id):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)


def create_data(config):
    json_decoder = json.JSONDecoder(object_pairs_hook=OrderedDict)

    with open(config.task_train_files, 'r') as _df:
        # data_info = json.load(_df)
        data_info = json_decoder.decode(_df.read())
    
    data_files = []
    counts = []
    # for fname in sorted(data_info.keys()):
    for fname in data_info.keys():
        count = data_info[fname]
        if config.data_dir != '':
            basename = os.path.basename(fname)
            full_fname = os.path.join(config.data_dir, basename)
        else:
            full_fname = fname
        data_files.append(full_fname)
        counts.append(int(count))
    
    if len(data_files) < hvd.size() - config.num_supervised_shards:
        # we should ideally not be in this scenario
        print("Warning: over-sampling as num shards is more than num files")
        data_files += np.random.choice(data_files, hvd.size() - config.num_supervised_shards - len(data_files), replace=False)
        counts = [int(data_info[os.path.basename(x)]) for x in data_files]
    counts = np.array(counts)

    print('Creating data')
    print("\n".join(["%s: %d" % (x, y) for x, y in zip(data_files, counts)]))

    sup_counts = np.array([])
    
    if not config.supervised_train_files is None and hvd.rank() < config.num_supervised_shards:
        assert hvd.size() > config.num_supervised_shards
        with open(config.supervised_train_files, 'r') as _df:
            # sup_data_info = json.load(_df)
            sup_data_info = json_decoder.decode(_df.read())
        sup_data_files = []
        sup_counts = []
        # for fname in sorted(sup_data_info.keys()):
        for fname in sup_data_info.keys():
            count = sup_data_info[fname]
            # basename = os.path.basename(fname)
            # full_fname = os.path.join(config.data_dir, basename)
            full_fname = fname
            sup_data_files.append(full_fname)
            sup_counts.append(int(count))
        sup_counts = np.array(sup_counts)
        print("Supervised data:")
        print("\n".join(["%s: %d" % (x, y) for x, y in zip(sup_data_files, sup_counts)]))
        sampling_weights = sup_counts ** 0.5 / np.sum(sup_counts ** 0.5)
        num_files_per_shard = int(len(sup_data_files) / config.num_supervised_shards)
        # first num_supervised_shards will get supervised data
        shard_start = hvd.rank() * num_files_per_shard
        shard_end = (hvd.rank() + 1) * num_files_per_shard
        if hvd.rank() == config.num_supervised_shards - 1:
            shard_files = sup_data_files[shard_start:]
            shard_weights = sampling_weights[shard_start:]
        else:
            shard_files = sup_data_files[shard_start: shard_end]
            shard_weights = sampling_weights[shard_start:shard_end]
        shard_weights = shard_weights / shard_weights.sum()
        print("Current worker %d files:" % len(shard_files), shard_files, shard_weights)
    else:
        # sampling_weights = counts ** 0.5 / sum(counts ** 0.5)
        sampling_weights = counts / counts.sum()
        num_shards = hvd.size() - config.num_supervised_shards
        num_files_per_shard = int(len(data_files) / num_shards)
        relative_rank = hvd.rank() - config.num_supervised_shards
        shard_start = relative_rank * num_files_per_shard
        shard_end = (relative_rank + 1) * num_files_per_shard
        if relative_rank == hvd.size() - 1:
            shard_files = data_files[shard_start:]
            shard_weights = sampling_weights[shard_start:]
        else:
            shard_files = data_files[shard_start: shard_end]
            shard_weights = sampling_weights[shard_start:shard_end]
        shard_weights = shard_weights / shard_weights.sum()
        print("Current worker %d files:" % len(shard_files), shard_files, shard_weights)

    if config.target_vocab != '':
        dataset = get_dataset(shard_files, shard_weights, config.train_batch_size, config.seq_length, ori_vocab=config.vocab, target_vocab=config.target_vocab, do_lower_case=config.do_lower_case)    
    else:
        dataset = get_dataset(shard_files, shard_weights, config.train_batch_size, config.seq_length)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = dataset.prefetch(32*20)
    return dataset, counts, sup_counts


def main(config):
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    dataset, counts, _ = create_data(config)

    checkpoint_dir = config.output_dir
    write_checkpoint_dir = checkpoint_dir
    write_summary_dir = os.path.join(write_checkpoint_dir, 'summaries')
    train_summary_writer = tf.summary.create_file_writer(write_summary_dir)

    model = MAMLFineTuner(config)
    
    if config.num_train_steps > 0:
        num_train_steps = config.num_train_steps
    else:
        num_train_steps = int(counts.sum() / (config.train_batch_size * (hvd.size() - config.num_supervised_shards)))
    warmup_steps = int(config.warmup_proportion*num_train_steps)
    print("Total steps: %d   Warmup steps: %d" % (num_train_steps, warmup_steps))
    optimizer = optimization.create_optimizer(config.learning_rate, 
                                                num_train_steps=num_train_steps,
                                                num_warmup_steps=warmup_steps,
                                                end_lr=0.,
                                                optimizer_type='adamw')
    # model.compile(
    #     optimizer=optimizer,
    #     loss_metric=tf.keras.metrics.Mean(),
    #     train_metric=tf.keras.metrics.Mean(),
    #     loss_scale=strategy.num_replicas_in_sync
    #     # metrics=tf.keras.metrics.Mean()
    # )
    loss_metric=tf.keras.metrics.Mean()
    train_metric=tf.keras.metrics.Mean()
    
    checkpoint = tf.train.Checkpoint(
            model=model, optimizer=optimizer, global_step=optimizer.iterations)
    if config.restore_checkpoint and tf.train.latest_checkpoint(config.restore_checkpoint):
        # import ipdb; ipdb.set_trace()
        print("Restoring Checkpoint from", config.restore_checkpoint)
        checkpoint.restore(tf.train.latest_checkpoint(config.restore_checkpoint)).assert_existing_objects_matched()
        if config.reset_optimizer_iters:
            # import ipdb; ipdb.set_trace()
            optimizer.iterations.assign(0)

    @tf.function
    def training_step(inputs, first_batch):
        print(tf.executing_eagerly())
        # per_replica_losses = strategy.run(model.train_step, args=(input_ids, input_mask, segement_ids, num_labels, label_ids))
        with tf.GradientTape() as tape:
            # loss, query_acc, query_pred = self(input_ids, input_mask, segement_ids, num_labels, label_ids, training=True)
            loss, query_acc, query_pred = model(inputs, training=True)
            raw_loss = loss
            # Need to do compute average loss as apply_gradients sums per-replica gradients
            # if self.loss_scale:
            #     loss = loss / self.loss_scale
        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)
        if config.adapter_hidden_size > 0:
            trainable_vars = list(filter(lambda x: 'adapters' in x.name or 'LayerNorm' in x.name, model.trainable_variables))
            if config.global_adapters and not config.learn_adapter_init:
                trainable_vars = list(filter(lambda x: 'global_adapters' in x.name, trainable_vars))
            all_learning_rates = []
            for _key in model.bert_learning_rates.keys():
                all_learning_rates += model.bert_learning_rates[_key]
            for _key in model.task_learning_rates.keys():
                all_learning_rates += model.task_learning_rates[_key]
            trainable_vars += all_learning_rates
            trainable_vars += list(model.task_weights.values())
        else:
            trainable_vars = model.trainable_variables # + list(self.bert_model.weights.values())
        print("Trainable vars:")
        print("\n".join([x.name for x in trainable_vars]))
        print("Total traininable params:", sum([np.prod(v.shape) for v in trainable_vars]))
        gradients = tape.gradient(loss, trainable_vars)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
        optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        #
        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        # tf.print('Rank:', hvd.rank(), 'loss:', raw_loss, 'label_emb_lr', model.task_learning_rates['label_embs'][0])
        if first_batch:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)
        
        # all_replica_loss = hvd.allgather(raw_loss)
        # all_replica_acc = hvd.allgather(query_acc)
        # print(all_replica_acc, all_replica_loss)
        if hvd.rank() == 0:
            # loss_metric.update_state(all_replica_loss.mean())
            # train_metric.update_state(all_replica_acc.mean())
            with train_summary_writer.as_default():
                for key, lr in model.bert_learning_rates.items():
                    tf.summary.scalar('layer_'+key, lr[0], step=optimizer.iterations)
                for key, lr in model.task_learning_rates.items():
                    tf.summary.scalar(key, lr[0], step=optimizer.iterations)
        # results = {m.name: m.result() for m in self.metrics}
        # results['query_acc'] = query_acc
        # acc = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_accs, axis=None)
        return loss, query_acc

    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=write_checkpoint_dir, max_to_keep=5)

    # import ipdb; ipdb.set_trace()
    # model.fit(dataset, epochs=config.num_epochs, steps_per_epoch=100)
    loss_metric = tf.keras.metrics.Mean()
    start_time = time.time()
    # for dist_inputs in dist_iterator:
    for step, inputs in enumerate(dataset.take(num_train_steps)):
        # input_ids, input_mask, segement_ids, num_labels, label_ids = next(dist_iterator)
        # input_ids, input_mask, segement_ids, num_labels, label_ids = dist_iterator.get_next()
        # print(tf.executing_eagerly())
        # step_loss = distributed_train_step(input_ids, input_mask, segement_ids, num_labels, label_ids)
        step_loss, step_acc = training_step(inputs, step == 0)
        # if hvd.rank() == 0:
        loss_metric.update_state(step_loss)
        train_metric.update_state(step_acc)
        # if hvd.rank() == 0:
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss_metric.result(), step=optimizer.iterations)
            tf.summary.scalar('query_accuracy', train_metric.result(), step=optimizer.iterations)
        step_time = time.time() - start_time
        sys.stdout.flush()
        optimizer_steps = optimizer.iterations.numpy()
        if optimizer_steps % config.save_steps == 0 and hvd.rank() == 0:
            tf.compat.v1.logging.info("Saving checkpoint...")
            checkpoint_manager.save()
            # import ipdb; ipdb.set_trace()
        if optimizer_steps % config.log_steps == 0:
            tf.compat.v1.logging.info("Step: " + str(optimizer_steps) +
                  ", loss: " + str(loss_metric.result().numpy()) +
                  ", worker_loss: " + str(step_loss.numpy()) +
                  ", query_acc: " + str(train_metric.result().numpy()) +
                  ", time: " + str(step_time))
            
            # wandb.log(
            #        {
            #            "loss": loss_metric.result().numpy(),
            #            "worker_loss": step_loss.numpy(),
            #            "query_acc": train_metric.result().numpy(),
            #            "time": step_time,
            #            }, step=optimizer_steps)
            # wandb.log(
            #        {'lr/layer_'+ key: lr[0].numpy()
            #            for key, lr in model.bert_learning_rates.items()
            #            }, step=optimizer_steps)
            # wandb.log(
            #        {'lr/' + key: lr[0].numpy()
            #            for key, lr in model.task_learning_rates.items()
            #            }, step=optimizer_steps)
        start_time = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test finetune environment')
    parser = command_line_args(parser)
    parser.add_argument('--num_epochs', default=5, type=int, help="number of GPU nodes")
    parser.add_argument('--meta_batchsz', default=80, type=int, help="support size for training")
    parser.add_argument('--data_dir', required=True, type=str, help="pre-training data folder")
    parser.add_argument('--task_train_files', required=True, type=str, 
                        help="json for unsupervised tasks")
    parser.add_argument('--supervised_train_files', default=None, type=str, 
                        help="json for supervised files")
    parser.add_argument('--num_supervised_shards', default=0, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--save_steps', type=int, default=10)
    parser.add_argument('--warmup_proportion', type=float, default=0.1, 
                        help="Proportion of training to perform linear learning rate warmup for. ")
    parser.add_argument('--log_steps', type=int, default=100)
    parser.add_argument('--restore_checkpoint', default='', type=str, 
                        help="full checkpoint to restore training")
    parser.add_argument('--reset_optimizer_iters', default=True, 
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--wandb_proj', type=str, default='metanlp')
    parser.add_argument('--wandb_run', type=str, default='default_run')
    parser.add_argument('--vocab', required=False, type=str, default='',
                        help="vocab of tf_record data (only provide if need to convert to another vocab)")
    parser.add_argument('--target_vocab', required=False, type=str, default='',
                        help="vocab to convert data (only provide if need to convert to another vocab)")
    parser.add_argument('--do_lower_case', default=False, 
                        type=lambda x: (str(x).lower() == 'true'),
                        help="only provide if need to convert to another vocab")
    parser.add_argument('--num_train_steps', default=-1, type=int, help='number of training steps to run')

    config = parser.parse_args()
    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    print("TF Version:", tf.__version__)
    # devices = tf.config.list_physical_devices('GPU')
    # for device in devices:
    #     tf.config.experimental.set_memory_growth(device, True)

    os.environ['NCCL_DEBUG'] = 'INFO'

    if hvd.rank() == 0:
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)

    print("WORLD SIZE:", hvd.size())
    main(config)
