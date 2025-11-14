from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner, SizePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFaceBaseModel
from opencompass.models import PyramidKVModel

with read_base():

    from opencompass.configs.datasets.longbench.longbenchnarrativeqa.longbench_narrativeqa_gen import LongBench_narrativeqa_datasets
    from opencompass.configs.datasets.longbench.longbenchqasper.longbench_qasper_gen import LongBench_qasper_datasets
    from opencompass.configs.datasets.longbench.longbenchmultifieldqa_en.longbench_multifieldqa_en_gen import LongBench_multifieldqa_en_datasets
    # from opencompass.configs.datasets.longbench.longbenchmultifieldqa_zh.longbench_multifieldqa_zh_gen import LongBench_multifieldqa_zh_datasets

    from opencompass.configs.datasets.longbench.longbenchhotpotqa.longbench_hotpotqa_gen import LongBench_hotpotqa_datasets
    from opencompass.configs.datasets.longbench.longbench2wikimqa.longbench_2wikimqa_gen import LongBench_2wikimqa_datasets
    from opencompass.configs.datasets.longbench.longbenchmusique.longbench_musique_gen import LongBench_musique_datasets
    # from opencompass.configs.datasets.longbench.longbenchdureader.longbench_dureader_gen import LongBench_dureader_datasets

    from opencompass.configs.datasets.longbench.longbenchgov_report.longbench_gov_report_gen import LongBench_gov_report_datasets
    from opencompass.configs.datasets.longbench.longbenchqmsum.longbench_qmsum_gen import LongBench_qmsum_datasets
    from opencompass.configs.datasets.longbench.longbenchmulti_news.longbench_multi_news_gen import LongBench_multi_news_datasets
    # from opencompass.configs.datasets.longbench.longbenchvcsum.longbench_vcsum_gen import LongBench_vcsum_datasets

    from opencompass.configs.datasets.longbench.longbenchtrec.longbench_trec_gen import LongBench_trec_datasets
    from opencompass.configs.datasets.longbench.longbenchtriviaqa.longbench_triviaqa_gen import LongBench_triviaqa_datasets
    from opencompass.configs.datasets.longbench.longbenchsamsum.longbench_samsum_gen import LongBench_samsum_datasets
    # from opencompass.configs.datasets.longbench.longbenchlsht.longbench_lsht_gen import LongBench_lsht_datasets

    from opencompass.configs.datasets.longbench.longbenchpassage_count.longbench_passage_count_gen import LongBench_passage_count_datasets
    from opencompass.configs.datasets.longbench.longbenchpassage_retrieval_en.longbench_passage_retrieval_en_gen import LongBench_passage_retrieval_en_datasets
    # from opencompass.configs.datasets.longbench.longbenchpassage_retrieval_zh.longbench_passage_retrieval_zh_gen import LongBench_passage_retrieval_zh_datasets

    from opencompass.configs.datasets.longbench.longbenchlcc.longbench_lcc_gen import LongBench_lcc_datasets
    from opencompass.configs.datasets.longbench.longbenchrepobench.longbench_repobench_gen import LongBench_repobench_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = [
    dict(
        abbr='llama3_1_8b-pyramidsnapkv-longbench',
        type=PyramidKVModel,
        path="meta-llama/Llama-3.1-8B",
        max_out_len=500,
        max_seq_len=32768,
        batch_size=1,
        generation_kwargs=dict(),
        method="snapkv",  # Set to pyramidkv method
        attn_implementation="flash_attention_2",  # You can choose other implementations as needed
        max_capacity_prompt=2052,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr='llama3_2_3b-pyramidsnapkv-longbench-512midstates',
        type=PyramidKVModel,
        path="meta-llama/Llama-3.2-3B",
        max_out_len=500,
        max_seq_len=32768,
        batch_size=1,
        generation_kwargs=dict(),
        method="snapkv",  # Set to pyramidkv method
        attn_implementation="flash_attention_2",  # You can choose other implementations as needed
        max_capacity_prompt=1540,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr='llama3_2_3b-pyramidsnapkv-longbench-1024midstates',
        type=PyramidKVModel,
        path="meta-llama/Llama-3.2-3B",
        max_out_len=500,
        max_seq_len=32768,
        batch_size=1,
        generation_kwargs=dict(),
        method="snapkv",  # Set to pyramidkv method
        attn_implementation="flash_attention_2",  # You can choose other implementations as needed
        max_capacity_prompt=2052,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr='llama3_2_3b-pyramidsnapkv-longbench-2048midstates',
        type=PyramidKVModel,
        path="meta-llama/Llama-3.2-3B",
        max_out_len=500,
        max_seq_len=32768,
        batch_size=1,
        generation_kwargs=dict(),
        method="snapkv",  # Set to pyramidkv method
        attn_implementation="flash_attention_2",  # You can choose other implementations as needed
        max_capacity_prompt=3076,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

work_dir = './outputs/pyramidsnapkv-longbench/'


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
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
