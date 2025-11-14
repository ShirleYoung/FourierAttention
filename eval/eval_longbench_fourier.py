from opencompass.models import FourierModel
from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

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
        abbr = 'llama3_1_8b-fourier-longbench',
        type=FourierModel,
        path="meta-llama/Llama-3.1-8B",
        model_type='llama',
        max_out_len=500,
        max_seq_len=32768,
        batch_size=1,
        generation_kwargs=dict(),
        non_critical_dims_path=" ",  #your path to "/FourierAttention/compressed_dims_file/llama3.1-8b/compressed_dims_32k_1024states_splithead.json"
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr = 'llama3_2_3b-fourierhippo-longbench-1024',
        type=FourierModel,
        path="meta-llama/Llama-3.2-3B",
        model_type='llama',
        max_out_len=500,
        max_seq_len=32768,
        batch_size=1,
        generation_kwargs=dict(),
        non_critical_dims_path=" ", #your path to "/FourierAttention/compressed_dims_file/llama3.2-3b/compressed_dims_32k_1024states_splithead.json" 
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr = 'llama3_2_3b-fourierhippo-longbench-512',
        type=FourierModel,
        path="meta-llama/Llama-3.2-3B",
        model_type='llama',
        max_out_len=500,
        max_seq_len=32768,
        batch_size=1,
        generation_kwargs=dict(),
        midstates=512,
        non_critical_dims_path=" ", #your path to "/FourierAttention/compressed_dims_file/llama3.2-3b/compressed_dims_32k_512states_splithead.json"
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        abbr = 'llama3_2_3b-fourierhippo-longbench-2048',
        type=FourierModel,
        path="meta-llama/Llama-3.2-3B",
        model_type='llama',
        max_out_len=500,
        max_seq_len=32768,
        batch_size=1,
        generation_kwargs=dict(),
        non_critical_dims_path=" ", #your path to "/FourierAttention/compressed_dims_file/llama3.2-3b/compressed_dims_32k_2048states_splithead.json"
        midstates=2048,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

work_dir = './outputs/fourier-longbench'

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