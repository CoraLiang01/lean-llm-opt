#### Index Sets

- $I$: set of all pizza types (indexed by $i$), corresponding to all unique values in **PizzaSalesDataset.csv**, column **Product Name**.

#### Parameters

- $A_i$: revenue per unit of pizza type $i$ (**PizzaSalesDataset.csv**, column **Revenue**).
- $d_i$: total demand for pizza type $i$ (**PizzaSalesDataset.csv**, column **Demand**).
- $I_i$: initial inventory for pizza type $i$ (**PizzaSalesDataset.csv**, column **Initial Inventory**).

#### Decision Variables

- $x_i$: number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \, x_i
$$

#### Constraints

1. **Inventory Constraints**  
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$

2. **Demand Constraints**  
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

3. **Non-negativity and Integrality**  
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **PizzaSalesDataset.csv**, column **Product Name** $\rightarrow$ index set $I$
- **PizzaSalesDataset.csv**, column **Revenue** $\rightarrow$ parameter $A_i$
- **PizzaSalesDataset.csv**, column **Demand** $\rightarrow$ parameter $d_i$
- **PizzaSalesDataset.csv**, column **Initial Inventory** $\rightarrow$ parameter $I_i$