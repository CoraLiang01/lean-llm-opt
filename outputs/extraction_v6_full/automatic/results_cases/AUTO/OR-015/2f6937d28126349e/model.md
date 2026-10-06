#### Index Sets

- $I$: Set of all products classified under ‘Aalop’, indexed by $i$.

#### Parameters

- $p_i$: Revenue per unit of product $i \in I$.  
  [CSVQA_DATA: file_0_view_0, column: Revenue, index: Product Name]
- $d_i$: Demand for product $i \in I$ over the sales horizon.  
  [CSVQA_DATA: file_0_view_0, column: Demand, index: Product Name]
- $s_i$: Initial inventory of product $i \in I$.  
  [CSVQA_DATA: file_0_view_0, column: Initial Inventory, index: Product Name]

#### Decision Variables

- $x_i$: Number of units of product $i \in I$ to fulfill (integer, $x_i \geq 0$).

#### Objective

$$
\max \sum_{i \in I} p_i x_i
$$

#### Constraints

1. Inventory constraint:
   $$
   x_i \leq s_i \qquad \forall i \in I
   $$
2. Demand constraint:
   $$
   x_i \leq d_i \qquad \forall i \in I
   $$
3. Nonnegativity and integrality:
   $$
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   $$

---

#### Data Mapping

- $I$: All records in [file_0_view_0] where [Product Name] starts with "Aalop".
- $p_i$: [Revenue] column in [file_0_view_0], indexed by [Product Name].
- $d_i$: [Demand] column in [file_0_view_0], indexed by [Product Name].
- $s_i$: [Initial Inventory] column in [file_0_view_0], indexed by [Product Name].