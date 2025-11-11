from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFaceBaseModel
from opencompass.models import StreamingModel

with read_base():
    from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_origin_en_datasets
    from opencompass.configs.summarizers.needlebench import needlebench_summarizer as summarizer

datasets = []
datasets += needlebench_origin_en_datasets

models = [
    dict(
        abbr = 'llama3.2_3b-streamingllm-needlebench',
        type = StreamingModel,
        path="meta-llama/Llama-3.2-3B",
        # model_type='llama',
        max_out_len=50,
        max_seq_len=32768,
        batch_size=1,
        generation_kwargs=dict(),
        enable_streaming = True,
        start_size = 4,
        recent_size = 1024,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr = 'llama3.1_8b-streamingllm-needlebench',
        type = StreamingModel,
        path="meta-llama/Llama-3.1-8B",
        # model_type='llama',
        max_out_len=50,
        max_seq_len=32768,
        batch_size=1,
        generation_kwargs=dict(),
        enable_streaming = True,
        start_size = 4,
        recent_size = 1024,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

work_dir = './outputs/streamingllm-needlebench/'

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
