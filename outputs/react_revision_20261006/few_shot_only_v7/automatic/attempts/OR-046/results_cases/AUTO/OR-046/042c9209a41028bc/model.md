##### Mathematical Model

Let:
- $I$ = set of products, indexed by $i$
- $x_i$ = number of units of product $i$ to order each day (decision variable)

Parameters:
- $v_i$ = value (benefit) per unit of product $i$
- $w_i$ = weight (space requirement) per unit of product $i$
- $C$ = total stock capacity

Objective:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

##### Data Mapping

- $I$: All records in `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv`, column `ProductName`
- $v_i$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv`, column `Value`, keyed by `ProductName`
- $w_i$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv`, column `Weight`, keyed by `ProductName`
- $C$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv`, column `Capacity`