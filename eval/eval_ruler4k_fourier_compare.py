from opencompass.models import FourierModel
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets
    # from opencompass.configs.datasets.longbench.longbench import longbench_datasets

datasets = ruler_datasets
# print("-"*50)
# print(f"datasets[0]:{datasets[0]}")
# print("-"*50)

models = [
    dict(
        abbr = 'llama3_2_3b-fourier-ruler4k-uniform',
        type=FourierModel,
        path="meta-llama/Llama-3.2-3B",
        model_type='llama',
        max_out_len=100,
        max_seq_len=4096,
        batch_size=1,
        generation_kwargs=dict(),
        non_critical_dims_path=" ", #your path to "/FourierAttention/compressed_dims_file/llama3.2-3b/compressed_dims_4k_uniform_splithead.json"
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr = 'llama3_2_3b-fourier-ruler4k-fourier',
        type=FourierModel,
        path="meta-llama/Llama-3.2-3B",
        model_type='llama',
        max_out_len=100,
        max_seq_len=4096,
        batch_size=1,
        generation_kwargs=dict(),
        non_critical_dims_path=" ", #your path to "/FourierAttention/compressed_dims_file/llama3.2-3b/compressed_dims_4k_fourier_splithead.json"
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr = 'llama3_2_3b-fourier-ruler-kvinverse',
        type=FourierModel,
        path="meta-llama/Llama-3.2-3B",
        model_type='llama',
        max_out_len=100,
        max_seq_len=4096,
        batch_size=1,
        generation_kwargs=dict(),
        non_critical_dims_path=" ", #your path to "/FourierAttention/compressed_dims_file/llama3.2-3b/compressed_dims_4k_kv-inverse_splithead.json"
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr = 'llama3_2_3b-fourier-ruler-layerinverse',
        type=FourierModel,
        path="meta-llama/Llama-3.2-3B",
        model_type='llama',
        max_out_len=100,
        max_seq_len=4096,
        batch_size=1,
        generation_kwargs=dict(),
        non_critical_dims_path=" ", #your path to "/FourierAttention/compressed_dims_file/llama3.2-3b/compressed_dims_4k_layer-inverse_splithead.json"
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

work_dir ='outputs/llama3_2_3b-fourier-ruler4k-ablation'

infer = dict(
    partitioner=dict(type=NaivePartitioner),  
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask), 
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)
