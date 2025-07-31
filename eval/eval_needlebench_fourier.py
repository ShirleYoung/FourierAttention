from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFaceBaseModel
from opencompass.models import HippoattnCausalLM

with read_base():
    from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_origin_en_datasets
    # from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_parallel_en_datasets
    from opencompass.configs.summarizers.needlebench import needlebench_summarizer as summarizer

datasets = []
datasets += needlebench_origin_en_datasets

is_single_niah = (len([key for key in list(locals()) if key.__contains__('parallel') and key.endswith('datasets')]) == 0)

models = [
    dict(
        abbr = 'llama3.1_8b-fourierhippo-needlebench',
        type=HippoattnCausalLM,
        path="meta-llama/Llama-3.1-8B",
        model_type='llama',
        max_out_len=50 if is_single_niah else 250,
        max_seq_len=409600,
        batch_size=1,
        generation_kwargs=dict(maxlocallen=1024, maxmidstates=1024, numinittokens=4, non_critical_dims_path=" "), #your path to "/FourierAttention/compressed_dims_file/llama3.1-8b/compressed_dims_32k_1024states.json"
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr = 'llama3.2_3b-fourierhippo-needlebench',
        type=HippoattnCausalLM,
        path="meta-llama/Llama-3.2-3B",
        model_type='llama',
        max_out_len=50 if is_single_niah else 250,
        max_seq_len=409600,
        batch_size=1,
        generation_kwargs=dict(maxlocallen=1024, maxmidstates=1024, numinittokens=4, non_critical_dims_path=" "), #your path to "/FourierAttention/compressed_dims_file/llama3.2-3b/compressed_dims_32k_1024states.json"
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

work_dir = './outputs/fourierhippo-needlebench/'



infer = dict(
    partitioner=dict(type=NaivePartitioner),  # dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(
        type=LocalRunner,
        # max_num_workers=2, 
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


