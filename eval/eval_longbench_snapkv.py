from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner, SizePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFaceBaseModel
from opencompass.models import PyramidKVModel

with read_base():
    from opencompass.configs.datasets.longbench.longbench import longbench_datasets

datasets = []
datasets += longbench_datasets


models = [
    dict(
        abbr='llama3.1_8b-pyramidsnapkv-longbench',
        type=PyramidKVModel,
        path="meta-llama/Llama-3.1-8B",
        max_out_len=500,
        max_seq_len=32768,
        batch_size=1,
        generation_kwargs=dict(),
        method="snapkv",  # 设定为 pyramidkv 方法
        attn_implementation="flash_attention_2",  # 您可以根据需要选择其他的实现方式
        max_capacity_prompt=2052,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr='llama3.2_3b-pyramidsnapkv-longbench-512midstates',
        type=PyramidKVModel,
        path="meta-llama/Llama-3.2-3B",
        max_out_len=500,
        max_seq_len=32768,
        batch_size=1,
        generation_kwargs=dict(),
        method="snapkv",  # 设定为 pyramidkv 方法
        attn_implementation="flash_attention_2",  # 您可以根据需要选择其他的实现方式
        max_capacity_prompt=1540,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr='llama3.2_3b-pyramidsnapkv-longbench-1024midstates',
        type=PyramidKVModel,
        path="meta-llama/Llama-3.2-3B",
        max_out_len=500,
        max_seq_len=32768,
        batch_size=1,
        generation_kwargs=dict(),
        method="snapkv",  # 设定为 pyramidkv 方法
        attn_implementation="flash_attention_2",  # 您可以根据需要选择其他的实现方式
        max_capacity_prompt=2052,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr='llama3.2_3b-pyramidsnapkv-longbench-2048midstates',
        type=PyramidKVModel,
        path="meta-llama/Llama-3.2-3B",
        max_out_len=500,
        max_seq_len=32768,
        batch_size=1,
        generation_kwargs=dict(),
        method="snapkv",  # 设定为 pyramidkv 方法
        attn_implementation="flash_attention_2",  # 您可以根据需要选择其他的实现方式
        max_capacity_prompt=3076,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

work_dir = './outputs/llama3.2_3b-pyramidsnapkv-longbench/'


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
# infer = dict(
#     partitioner=dict(type=NaivePartitioner),  # dict(type=NumWorkerPartitioner, num_worker=4),
#     runner=dict(
#         type=LocalRunner,
#         # max_num_workers=1, 
#         task=dict(type=OpenICLInferTask), 
#     ),
# )

# eval = dict(
#     partitioner=dict(type=NaivePartitioner),
#     runner=dict(
#         type=LocalRunner,
#         max_num_workers=16, 
#         task=dict(type=OpenICLEvalTask, dump_details=True),
#     ),
# )

# summarizer = dict(
#     dataset_abbrs=['ruler_4k', 'ruler_8k', 'ruler_16k', 'ruler_32k', 'ruler_128k'],
#     summary_groups=sum(
#         [v for k, v in locals().items() if k.endswith('_summary_groups')], []
#     ),
# )

# source /fs-computility/llm/liuxiaoran/.bashrc
# conda activate /cpfs01/user/liuxiaoran/miniconda3/envs/llm-cuda12.1
# python run.py eval_xrliu/eval_xrliu_niah.py --dump-eval-details --debug -r  调试用
# python run.py eval_xrliu/eval_xrliu_niah.py --dump-eval-details -r 20240820_190019 第一次用
