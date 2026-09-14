扰动实验 2：303 题交付包说明
================================

一、这个文件夹里有什么

1. test_set_303.csv
   正式测试表，共 303 行，不含 Gold。每行是一道扰动后的题目。

2. datasets/
   每道题对应的数据文件。共有 303 个子文件夹、546 个 CSV 文件。
   test_set_303.csv 的 Dataset_address 列给出该题要读取的一个或多个 CSV；
   多个文件路径用换行分隔，路径均相对于本交付文件夹。

3. gold_303.csv
   303 道题的标准答案。通过 variant_id 与 test_set_303.csv 一一对应。
   Label-objective 是标准最优目标值，Label-model 是标准 Gurobi LP 模型。
   推理结束前不得把此文件、其中的目标值或模型提供给被测试模型。


二、三套 seed 是什么

- 原始题目数：101。
- 数据 seed：17、42、73。
- 每个 seed 都对同一批 101 道原题生成一套独立、可复现的扰动，因此总数是
  101 × 3 = 303。
- 每个 seed 都恰好有 101 行；它们不是难度等级，也不是训练集、验证集、测试集划分。
- data_seed 只决定“数据怎样扰动”，不等于模型推理时使用的随机种子。
- 题目编号格式为 additive__<instance_id>__seed<seed>，例如：
  additive__TP1__seed17。


三、扰动设定

对每个语义上可变的原始业务参数，同时扰动目标函数参数和约束参数：

    新值 = 原值 + d，d ∈ {-2, -1, +1, +2}

具体规则：

- 每个参数的 d 由 seed、instance_id、parameter_key 和重采样次数共同通过
  SHA256 确定，所以同一版本重复生成可复现，不同 seed 会形成不同组合。
- 价格、成本、利润、需求、供给、容量、上下界、资源消耗、加工时间、业务比例
  等语义可变数字可以扰动。
- 结构常数、变量类型、约束形式、结构零和定义性常数不扰动。
- Query 和 CSV 中表示同一业务量的数字使用同一个 offset，不能各自独立修改。
- 整数保持整数；小数保持原有精度规则；非法负需求、负容量、越界比例等方向会被过滤。
- 派生值不独立抽取 offset，而是按扰动后的原始值重新计算。
- Mixture17 使用工作日守恒转移：每列总和保持 10，直接变化仍只允许 ±1 或 ±2。
- 若候选题不可行、无有限最优解，或与同题另一 seed 完全重复，则按确定性规则重采样。

本交付包对应修订版 release：
perturbation_exp2_additive_full101_schemafix_queryclarity_r2。
303 道题都已由 Gurobi 验证为有限最优 OPTIMAL。


四、用主实验 ipynb 测试

以下说明以当前主实验 notebook
LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or_V2_20260912.ipynb 为准。

1. 从项目根目录启动 Jupyter，顺序运行 notebook，直到 run_test 和 report_results
   已定义。不要先读取 gold_303.csv。

2. 新增一个单元格，粘贴并运行下面的代码。建议按 seed 分三次运行，便于断点续跑和检查：

    from pathlib import Path
    import pandas as pd

    DELIVERY_ROOT = (
        PROJECT_ROOT / "Test_Dataset/perturbation-large-scale-or"
    ).resolve()

    all_test = pd.read_csv(
        DELIVERY_ROOT / "test_set_303.csv",
        encoding="utf-8-sig",
    )

    assert len(all_test) == 303
    assert all_test["variant_id"].is_unique
    assert all_test.groupby("data_seed").size().to_dict() == {
        17: 101, 42: 101, 73: 101
    }

    def make_seed_test(seed):
        frame = all_test.loc[all_test["data_seed"] == seed].copy()
        frame["problem_id"] = frame["variant_id"]
        frame["Dataset_address"] = frame["Dataset_address"].map(
            lambda value: "\n".join(
                str((DELIVERY_ROOT / Path(line.strip())).resolve())
                for line in str(value).splitlines()
                if line.strip()
            )
        )
        return frame

    for seed in (17, 42, 73):
        seed_test = make_seed_test(seed)
        seed_output_dir = RESULTS_DIR / "perturbation_exp2_303" / f"seed{seed}"
        seed_results = run_test(
            seed_test,
            output_csv=seed_output_dir / "results.csv",
            routes=None,
        )
        report_results(seed_results, seed_output_dir)

这里把 problem_id 设为 variant_id，是为了让模型结果能够与 Gold 精确对齐。
这里也把 Dataset_address 转成绝对路径；若不做这一步，notebook 从项目根目录运行时
会错误地到项目根目录下寻找 datasets/。

3. 确认三个结果文件都各有 101 行：

    RESULTS_DIR/perturbation_exp2_303/seed17/results.csv
    RESULTS_DIR/perturbation_exp2_303/seed42/results.csv
    RESULTS_DIR/perturbation_exp2_303/seed73/results.csv

推理阶段只使用 test_set_303.csv 和 datasets/。三个 seed 全部完成并固定结果后，
才能打开 gold_303.csv 评分。


五、推理完成后评分

在三个 seed 全部跑完之后，再运行下面的单元格：

    import numpy as np
    import pandas as pd

    result_parts = []
    for seed in (17, 42, 73):
        part = pd.read_csv(
            RESULTS_DIR / "perturbation_exp2_303" / f"seed{seed}" / "results.csv"
        )
        part["variant_id"] = part["problem_id"]
        result_parts.append(part)

    results_303 = pd.concat(result_parts, ignore_index=True)
    gold_303 = pd.read_csv(
        DELIVERY_ROOT / "gold_303.csv",
        usecols=["variant_id", "gold_status", "objective_sense", "Label-objective"],
        encoding="utf-8-sig",
    )

    scored_303 = results_303.merge(
        gold_303,
        on="variant_id",
        how="left",
        validate="one_to_one",
    )
    actual = pd.to_numeric(scored_303["final_objective"], errors="coerce")
    expected = pd.to_numeric(scored_303["Label-objective"], errors="coerce")
    scored_303["objective_correct"] = np.isclose(
        actual, expected, rtol=1e-4, atol=1e-4, equal_nan=False
    )

    assert len(scored_303) == 303
    assert scored_303["variant_id"].is_unique
    print(
        f"正确 {int(scored_303['objective_correct'].sum())}/303，"
        f"准确率 {scored_303['objective_correct'].mean():.2%}"
    )
    scored_303.to_csv(
        RESULTS_DIR / "perturbation_exp2_303/scored_results_303.csv",
        index=False,
        encoding="utf-8-sig",
    )


六、最重要的三条注意事项

1. 不要在推理前把 test_set_303.csv 与 gold_303.csv 合并。
2. 不要把 Label-objective 或 Label-model 放进模型提示词、RAG、fallback 或选择逻辑。
3. 对比不同模型或机制时，必须使用同一批 303 个 variant_id、相同顺序和相同评分容差。
