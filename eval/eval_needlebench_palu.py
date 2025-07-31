from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner
from opencompass.tasks import OpenICLInferTask
from opencompass.models import PaluCausalLM

with read_base():
    from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_origin_en_datasets
    # from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_parallel_en_datasets
    from opencompass.configs.summarizers.needlebench import needlebench_summarizer as summarizer

datasets = []
datasets += needlebench_origin_en_datasets
is_single_niah = (len([key for key in list(locals()) if key.__contains__('parallel') and key.endswith('datasets')]) == 0)
models = [
    dict(
        abbr='llama3_1_8b-palu-needlebench',
        type=PaluCausalLM,
        path='/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/projects_402/Palu/Llama3_1_8b_ratio-0.7_gs-4-fisher_uniform-whiten',
        max_out_len=50 if is_single_niah else 250,
        max_seq_len=409600,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

work_dir = './outputs/llama3_1_8b-palu-needlebench/'

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
    ),
)
