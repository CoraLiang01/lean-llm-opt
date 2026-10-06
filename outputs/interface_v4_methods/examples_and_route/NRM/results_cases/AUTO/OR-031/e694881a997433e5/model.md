#### Abstract Mathematical Model

Let:
- $I$ = set of dairy products, indexed by $i$ (from DairyGoodsSalesDataset.csv, column Full_Product_Name)
- For each $i \in I$:
    - $r_i$ = revenue per unit of product $i$ (Revenue)
    - $d_i$ = demand for product $i$ (Demand)
    - $s_i$ = initial inventory of product $i$ (Initial Inventory)
- Decision variables: $x_i$ = number of units of product $i$ to be fulfilled

Objective:
$$
\max \sum_{i \in I} r_i x_i
$$

Subject to:
1. Inventory limit for each product:
$$
x_i \leq s_i \quad \forall i \in I
$$

2. Demand limit for each product:
$$
x_i \leq d_i \quad \forall i \in I
$$

3. Non-negativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

#### Data Mapping

- $I$: DairyGoodsSalesDataset.csv, column Full_Product_Name
- $r_i$: DairyGoodsSalesDataset.csv, column Revenue, for product $i$
- $d_i$: DairyGoodsSalesDataset.csv, column Demand, for product $i$
- $s_i$: DairyGoodsSalesDataset.csv, column Initial Inventory, for product $i$
- $x_i$: number of units of product $i$ to be fulfilled (decision variable, nonnegative integer)

All parameters and index sets are taken directly from DairyGoodsSalesDataset.csv, preserving source order and identifiers.