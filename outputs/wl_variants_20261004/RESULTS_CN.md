# 36 个变体与当前 RefData 的 WL 相似度

> 已补充用户提供的 questions.csv 类别映射。按变体真实类别汇总的最新结果见
> [CATEGORY_RESULTS_CN.md](/Users/cora/Documents/GitHub/lean-llm-opt/outputs/wl_variants_20261004/CATEGORY_RESULTS_CN.md)。
> 本文保留此前未提供类别时的全参考库分析；全库数值仍然有效。

计算日期：2026-10-04。使用原有 typed 变量—约束二部图、累计第 0 到 h 轮的 WL 标签计数及余弦归一化。主设置 h=2，同时输出 h=1、3。未预处理或求解模型，未修改输入文件。

输入为 `Variants_1_36_lp/Variants1.lp` 至 `Variants36.lp`，参考为 `Ref_Data_Large_Scale_LP/1.lp` 至 `15.lp`。参考类别取自当前 `Large_Scale_Or_Files/RAG_Examples_All.csv` 的对应数据行。参考 CSV 与全部参考 LP 的指纹均与 2026-10-03 记录一致。

**变体类别未提供，因此这里报告全参考库最近相似度，不将最近参考的类别当作变体的真实类别。**

对每个变体分别计算与 15 个参考的分数，取最高值 S_all，再对 36 个最高值统计。分数越高表示该图表示下越接近，不是准确率或相同约束的百分比。

|WL 深度|全库最近中位数|全库最近均值|
|---|---:|---:|
|1|0.7460|0.6756|
|2|0.5588|0.5373|
|3|0.4673|0.4511|

主结果 IQR 为 [0.3604, 0.7059]；范围 [0.3116, 0.7544]；36 个最高分均低于 1。IQR 为 25%—75% 分位数，不是置信区间。

|变体|最近参考|该参考类别|相似度 h=2|
|---|---|---|---:|
|Variants1|Ref10|Mixture|0.7478|
|Variants2|Ref10|Mixture|0.6644|
|Variants3|Ref04|Mixture|0.4475|
|Variants4|Ref04|Mixture|0.4627|
|Variants5|Ref01|AP|0.3116|
|Variants6|Ref01|AP|0.3129|
|Variants7|Ref12|Others|0.5490|
|Variants8|Ref04|Mixture|0.3374|
|Variants9|Ref14|RA|0.6865|
|Variants10|Ref13|Others|0.5685|
|Variants11|Ref10|Mixture|0.7544|
|Variants12|Ref10|Mixture|0.6609|
|Variants13|Ref06|Mixture|0.4503|
|Variants14|Ref04|Mixture|0.4381|
|Variants15|Ref01|AP|0.3121|
|Variants16|Ref01|AP|0.3124|
|Variants17|Ref04|Mixture|0.3382|
|Variants18|Ref09|Mixture|0.6079|
|Variants19|Ref13|Others|0.5740|
|Variants20|Ref04|Mixture|0.4405|
|Variants21|Ref04|Mixture|0.4466|
|Variants22|Ref01|AP|0.3780|
|Variants23|Ref10|Mixture|0.6628|
|Variants24|Ref04|Mixture|0.3374|
|Variants25|Ref01|AP|0.3569|
|Variants26|Ref01|AP|0.3615|
|Variants27|Ref01|AP|0.3563|
|Variants28|Ref04|Mixture|0.7069|
|Variants29|Ref04|Mixture|0.7095|
|Variants30|Ref04|Mixture|0.7057|
|Variants31|Ref04|Mixture|0.7126|
|Variants32|Ref04|Mixture|0.7055|
|Variants33|Ref04|Mixture|0.7067|
|Variants34|Ref04|Mixture|0.7398|
|Variants35|Ref04|Mixture|0.7402|
|Variants36|Ref04|Mixture|0.7392|

以上类别指的是最近参考类别，并非变体分类。没有同类别映射时，不能直接与此前的“同类别最近相似度”表比较。

此前 101 个原测试实例的全库最近中位数为 0.6097、均值为 0.5812；当前 36 个变体分别为 0.5588、0.5373。虽然当前汇总数值较小，两批样本的构成不同，且没有逐题原始—变体映射，不能据此归因于变体修改，也不能推断求解鲁棒性。

本方法忽略目标函数、系数数值/符号、右端项和上下界数值。单纯改变目标或数值而不改变图和类型，可能保持相同分数。变量域编码和规模变化会影响分数，不能将较低分直接解释为新的业务要求。

输出文件：
- `pairwise_h1.csv`、`pairwise_h2.csv`、`pairwise_h3.csv`：完整 36×15 矩阵。
- `nearest_h1.csv`、`nearest_h2.csv`、`nearest_h3.csv`：逐题全库最近结果、并列参考，以及对各参考类别的最大分数。
- `reference_pool_summary.csv`：全部 36 个变体分别与某参考类别池匹配后的汇总；不是按变体类别统计。
- `manifest.csv`：51 个 LP 的路径、SHA-256、规模与类型。
- `audit.json`：参数、版本核查和实现指纹。
- `recompute_variants_wl.py`：调用项目根目录 `wl_bipartite.py` 的复现入口。

复现命令（仓库根目录）：
```sh
/usr/bin/python3 outputs/wl_variants_20261004/recompute_variants_wl.py
```
