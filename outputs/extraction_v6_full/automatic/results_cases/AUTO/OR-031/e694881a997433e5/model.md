#### Abstract Mathematical Model

Let:

- $I$ = set of all dairy products, indexed by $i$
- $r_i$ = revenue per unit of product $i$ (parameter)
- $d_i$ = demand for product $i$ (parameter)
- $s_i$ = initial inventory for product $i$ (parameter)
- $x_i$ = number of units of product $i$ to fulfill (decision variable)

##### Variables

$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
$$

##### Objective

$$
\max \sum_{i \in I} r_i \cdot x_i
$$

##### Constraints

1. Inventory constraint:
   $$
   x_i \leq s_i, \quad \forall i \in I
   $$
2. Demand constraint:
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

##### Data Mapping

- $I$: All unique values in column Full_Product_Name from table_id file_0_view_0 in DairyGoodsSalesDataset.csv
- $r_i$: Revenue from column Revenue, indexed by Full_Product_Name, table_id file_0_view_0
- $d_i$: Demand from column Demand, indexed by Full_Product_Name, table_id file_0_view_0
- $s_i$: Initial Inventory from column Initial Inventory, indexed by Full_Product_Name, table_id file_0_view_0

(CSVQA_DATA bindings: file_0_view_0, columns Full_Product_Name, Revenue, Demand, Initial Inventory)