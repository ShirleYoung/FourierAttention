from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFaceBaseModel
from opencompass.models import StreamingModel

with read_base():
    # from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_origin_en_datasets
    # from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_parallel_en_datasets
    # from opencompass.configs.summarizers.needlebench import needlebench_summarizer as summarizer
    from opencompass.configs.datasets.longbench.longbench import longbench_datasets

datasets = []
datasets += longbench_datasets

# is_single_niah = (len([key for key in list(locals()) if key.__contains__('parallel') and key.endswith('datasets')]) == 0)


models = [
    dict(
        abbr = 'llama3.1_8b-streamingllm-longbench',
        type = StreamingModel,
        path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.1-8B/",
        # model_type='llama',
        max_out_len=100,
        max_seq_len=409600,
        batch_size=1,
        generation_kwargs=dict(),
        enable_streaming = True,
        start_size = 4,
        recent_size = 1024,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr = 'llama3.2_3b-streamingllm-longbench',
        type = StreamingModel,
        path="/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B/",
        # model_type='llama',
        max_out_len=100,
        max_seq_len=409600,
        batch_size=1,
        generation_kwargs=dict(),
        enable_streaming = True,
        start_size = 4,
        recent_size = 1024,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

work_dir = './outputs/streamingllm-longbench/'



infer = dict(
    partitioner=dict(type=NaivePartitioner),  # dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(
        type=LocalRunner,
        # max_num_workers=1, 
        task=dict(type=OpenICLInferTask), 
    ),
)

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
