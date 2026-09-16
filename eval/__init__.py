"""RAG 评测包.

模块划分::

    metrics.py              指标计算(纯函数, 无 IO, 可独立单测)
    common.py               工作区隔离、评测集读写、文档入库
    generate_golden_set.py  从文档自动生成评测题候选
    run_eval.py             评测运行器与报告生成
    golden_set.jsonl        评测集(按证据文本标注, 与分块策略解耦)
    results/                评测输出
"""
