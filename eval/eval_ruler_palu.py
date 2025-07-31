from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner
from opencompass.tasks import OpenICLInferTask
from opencompass.models import PaluCausalLM

with read_base():
    # from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_origin_en_datasets
    # from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_parallel_en_datasets
    # from opencompass.configs.summarizers.needlebench import needlebench_summarizer as summarizer
    from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets

datasets = []
datasets += ruler_datasets

models = [
    dict(
        abbr='llama3_2_3b-palu-ruler8k',
        type=PaluCausalLM,
        path='/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_402/Palu/Llama3_2_3b_ratio-0.7_gs-4-fisher_uniform-whiten',
        max_out_len=200,
        max_seq_len=409600,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

work_dir = './outputs/llama3_2_3b-palu-ruler8k/'

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
    ),
)
