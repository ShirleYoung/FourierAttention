from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import PaluCausalLM

with read_base():
    from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_origin_en_datasets
    from opencompass.configs.summarizers.needlebench import needlebench_summarizer as summarizer

datasets = []
datasets += needlebench_origin_en_datasets

models = [
    dict(
        abbr='llama3_1_8b-palu-needlebench',
        type=PaluCausalLM,
        path='',  # your path for llama3_1_8b-palu 
        max_out_len=50,
        max_seq_len=409600,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ), 
    dict(
        abbr='llama3_2_3b-palu-needlebench',
        type=PaluCausalLM,
        path='',  # your path for llama3_2_3b-palu 
        max_out_len=50,
        max_seq_len=409600,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ), 
]

work_dir = './outputs/llama3_1_8b-palu-needlebench/'

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
    ),
)
