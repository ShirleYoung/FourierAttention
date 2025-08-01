from opencompass.models import HippoattnCausalLM
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets

datasets = ruler_datasets

models = [
    dict(
        abbr = 'llama3_2-3b-hippo-legT-ruler4k',
        type=HippoattnCausalLM,
        path="meta-llama/Llama-3.2-3B",
        model_type='llama',
        max_out_len=100,
        max_seq_len=4096,
        batch_size=1,
        generation_kwargs=dict(maxlocallen=1024, maxmidstates=1024, numinittokens=4, non_critical_dims_path=" "), #your path to "/FourierAttention/compressed_dims_file/llama3.2-3b/compressed_dims_4k_legT.json"
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

work_dir ='outputs/llama3_2-3b-hippo-legT-ruler4k'

infer = dict(
    partitioner=dict(type=NaivePartitioner),  # dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(
        type=LocalRunner,
        # max_num_workers=2,
        task=dict(type=OpenICLInferTask)),
    # runner=dict(
    #     type=DLCRunner,
    #     max_num_workers=256,  # 84,
    #     task=dict(type=OpenICLInferTask),
    #     aliyun_cfg=aliyun_cfg, 
    #     preemptible=True, 
    #     priority=6, 
    #     retry=8),
)

# eval = dict(
#     partitioner=dict(type=NaivePartitioner),

#     # runner=dict(
#     #     type=DLCRunner,
#     #     max_num_workers=84,
#     #     task=dict(type=OpenICLEvalTask),
#     #     aliyun_cfg=aliyun_cfg,
#     #     preemptible=True, 
#     #     priority=9, 
#     #     retry=2),
# )