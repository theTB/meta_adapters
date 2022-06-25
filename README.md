## Dependencies
The code uses horovord library with tensorflow for multi-gpu training. 

Refer to `horovod_environment.yml` and `requirements.txt` for dependencies.

Cuda 10.1.168 is required.


### Creating environment. 
Make sure `cuda 10.1.168` is available. You can use miniconda to create a virtual environment as follows:

```bash
$ export ENV_PREFIX=$PWD/env
$ export HOROVOD_CUDA_HOME=$CUDA_ROOT
$ export HOROVOD_NCCL_HOME=$ENV_PREFIX
$ export HOROVOD_GPU_OPERATIONS=NCCL
$ export OMPI_MCA_opal_cuda_support=true
$ conda env create --prefix $ENV_PREFIX --file horovod_environment.yml --force
```

## Meta-Training
This requires large training time and typically should be run on multiple GPU.

Generate unsupervised meta-training tasks following https://github.com/iesl/metanlp and put in folder `data` along with a json listing the files (see data/data.json).

Sample data can be downloaded from: https://drive.google.com/file/d/18rL0rDF-zu6hxOSe1Fd15_O3tc6sXr28/view?usp=sharing

Save data in the folder `data` . 

Downlad BERT base cased model checkpoints and configs, and save in folder `bert`: https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip

Code has been test on 32G V100. Use at least 2 GPU to run the training.
Commad for training:
```bash
export CUDA_HOME=$CUDA_ROOT
conda activate $PWD/env
horovodrun --mpi-args='--oversubscribe' -np 2 -H localhost:2 python src/pretrain_finetuner_horovord.py --bert_config_file=bert/cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=bert/cased_L-12_H-768_A-12/bert_model.ckpt --data_dir=$PWD/data --task_train_files=data/data.json --supervised_train_files=data/supervised_data.json --num_supervised_shards=1 --inner_epochs=1 --average_query_every=3 --train_batch_size=90 --meta_batchsz=80 --num_batches=8 --stop_grad=true --output_dir=output/ --save_steps=5000 --log_steps=10 --learning_rate=1e-5 --train_lr=1e-5 --parameter_efficient=true --adapter_hidden_size=16 --global_adapters=true
```

## Fine-tuning
Download fine-tuning splits from https://github.com/iesl/metanlp 

After the model is trained, fine-tune as follows:
```bash
python src/finetune_eval.py --support_file=<tf_record> --test_file=<tf_record> --tf2_checkpoint=<trained_ckpt> --epochs=100 --train_batch_size=16 --num_labels=<num_labels> --bert_config_file=bert/cased_L-12_H-768_A-12/bert_config.json --k=<k> --parameter_efficient=true --global_adapters=true --adapter_hidden_size=16 --loss_threshold=1e-3
```

## Acknowledgements
Code uses Horovord library.

Code builds on existing open-source code https://github.com/google-research/bert