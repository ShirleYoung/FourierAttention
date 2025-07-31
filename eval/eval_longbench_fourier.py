from opencompass.models import HippoattnCausalLM
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from opencompass.configs.datasets.longbench.longbench import longbench_datasets

datasets = longbench_datasets
# print("-"*50)
# print(f"datasets[0]:{datasets[0]}")
# print("-"*50)

models = [
    dict(
        abbr = 'llama3_1-8b-fourierhippo-longbench',
        type=HippoattnCausalLM,
        path="meta-llama/Llama-3.1-8B",
        model_type='llama',
        max_out_len=500,
        max_seq_len=31500,
        batch_size=1,
        generation_kwargs=dict(maxlocallen=1024, maxmidstates=1024, numinittokens=4, non_critical_dims_path=" "),  #your path to "/FourierAttention/compressed_dims_file/llama3.1-8b/compressed_dims_32k_1024states.json"
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr = 'llama3_2-3b-fourierhippo-longbench-1024',
        type=HippoattnCausalLM,
        path="meta-llama/Llama-3.2-3B",
        model_type='llama',
        max_out_len=500,
        max_seq_len=31500,
        batch_size=1,
        generation_kwargs=dict(maxlocallen=1024, maxmidstates=1024, numinittokens=4, non_critical_dims_path=" "), #your path to "/FourierAttention/compressed_dims_file/llama3.2-3b/compressed_dims_32k_1024states.json"
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr = 'llama3_2-3b-fourierhippo-longbench-512',
        type=HippoattnCausalLM,
        path="meta-llama/Llama-3.2-3B",
        model_type='llama',
        max_out_len=500,
        max_seq_len=31500,
        batch_size=1,
        generation_kwargs=dict(maxlocallen=1024, maxmidstates=512, numinittokens=4, non_critical_dims_path=" "), #your path to "/FourierAttention/compressed_dims_file/llama3.2-3b/compressed_dims_32k_512states.json"
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr = 'llama3_2-3b-fourierhippo-longbench-2048',
        type=HippoattnCausalLM,
        path="meta-llama/Llama-3.2-3B",
        model_type='llama',
        max_out_len=500,
        max_seq_len=31500,
        batch_size=1,
        generation_kwargs=dict(maxlocallen=1024, maxmidstates=2048, numinittokens=4, non_critical_dims_path=" "), #your path to "/FourierAttention/compressed_dims_file/llama3.2-3b/compressed_dims_32k_2048states.json"
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

work_dir ='outputs/fourierhippo-longbench'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
    runner=dict(
        type=LocalRunner,
        # max_num_workers=4, retry=2, 
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16, retry=2, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)