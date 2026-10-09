ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of products, indexed by $i$

Parameters:
- $l_i$: labor required per unit of product $i$
- $m_i$: material required per unit of product $i$
- $s_i$: selling price per unit of product $i$
- $v_i$: variable cost per unit of product $i$
- $L$: total weekly labor capacity ($=1650$)
- $M$: total weekly material capacity ($=1850$)
- $F$: fixed weekly operating cost ($=4500$)

Decision Variables:
- $x_i \geq 0$: quantity of product $i$ to produce (continuous)

Objective:
\[
\max \left( \sum_{i \in I} (s_i - v_i) x_i - F \right)
\]

Subject to:
\[
\sum_{i \in I} l_i x_i \leq L
\]
\[
\sum_{i \in I} m_i x_i \leq M
\]
\[
x_i \geq 0 \quad \forall i \in I
\]

Data Mapping:

- $I$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv, column "Product Name"
- $l_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv, column "Labor per unit"
- $m_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv, column "Material per unit"
- $s_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv, column "Selling Price"
- $v_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv, column "Variable Cost"
- $L$: 1650 (from user description)
- $M$: 1850 (from user description)
- $F$: 4500 (from user description)