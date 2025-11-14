from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import PyramidKVModel

with read_base():
    from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_origin_en_datasets
    from opencompass.configs.summarizers.needlebench import needlebench_summarizer as summarizer

datasets = []
datasets += needlebench_origin_en_datasets

models = [
    dict(
        abbr='llama3_1_8b-pyramidsnapkv-needlebench',
        type=PyramidKVModel,
        path="meta-llama/Llama-3.1-8B",
        max_out_len=50,
        max_seq_len=409600,
        batch_size=1,
        generation_kwargs=dict(),
        method="snapkv",  # Set to pyramidkv method
        attn_implementation="flash_attention_2",  # You can choose other implementations as needed
        max_capacity_prompt=2052,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr='llama3_2_3b-pyramidsnapkv-needlebench',
        type=PyramidKVModel,
        path="meta-llama/Llama-3.2-3B",
        max_out_len=50,
        max_seq_len=409600,
        batch_size=1,
        generation_kwargs=dict(),
        method="snapkv",  # Set to pyramidkv method
        attn_implementation="flash_attention_2",  # You can choose other implementations as needed
        max_capacity_prompt=2052,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

work_dir = './outputs/pyramidsnapkv-needlebench/'

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
        max_num_workers=16, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)
