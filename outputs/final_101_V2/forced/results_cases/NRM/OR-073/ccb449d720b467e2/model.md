### Abstract Mathematical Model

#### Index Sets
- $P$: Set of products, indexed by $p$ (e.g., I, II, III)
- $A$: Set of equipment for procedure A, indexed by $a$ (e.g., A1, A2)
- $B$: Set of equipment for procedure B, indexed by $b$ (e.g., B1, B2, B3)
- $E$: Set of all equipment, $E = A \cup B$

#### Parameters
- $t_{a,p}$: Processing time (hours/unit) of product $p$ on equipment $a \in A$
- $t_{b,p}$: Processing time (hours/unit) of product $p$ on equipment $b \in B$
- $T_e$: Available operating time (hours) for equipment $e \in E$
- $C_e$: Equipment cost at full load (yuan) for equipment $e \in E$
- $c_p$: Raw material cost per unit of product $p$ (yuan/unit)
- $s_p$: Selling price per unit of product $p$ (yuan/unit)
- $\delta_{a,p}$: 1 if product $p$ can be processed on equipment $a$ for procedure A, 0 otherwise
- $\delta_{b,p}$: 1 if product $p$ can be processed on equipment $b$ for procedure B, 0 otherwise

#### Decision Variables
- $x_p \geq 0$: Production quantity of product $p$ (continuous)
- $y_{a,p} \geq 0$: Quantity of product $p$ processed on equipment $a$ for procedure A
- $z_{b,p} \geq 0$: Quantity of product $p$ processed on equipment $b$ for procedure B

#### Objective Function
\[
\max \left\{ \sum_{p \in P} \left[ s_p x_p - c_p x_p \right] - \sum_{e \in E} C_e \cdot \frac{1}{T_e} \cdot \left( \sum_{p \in P} t_{e,p} \cdot q_{e,p} \right) \right\}
\]
where $q_{e,p}$ is $y_{a,p}$ if $e=a \in A$, $z_{b,p}$ if $e=b \in B$, and $t_{e,p}$ is the corresponding processing time.

#### Constraints

1. **Production Assignment (Procedure A):**
   \[
   x_p = \sum_{a \in A} y_{a,p} \quad \forall p \in P
   \]
   \[
   y_{a,p} = 0 \quad \text{if } \delta_{a,p} = 0
   \]

2. **Production Assignment (Procedure B):**
   \[
   x_p = \sum_{b \in B} z_{b,p} \quad \forall p \in P
   \]
   \[
   z_{b,p} = 0 \quad \text{if } \delta_{b,p} = 0
   \]

3. **Equipment Time Capacity:**
   \[
   \sum_{p \in P} t_{e,p} \cdot q_{e,p} \leq T_e \quad \forall e \in E
   \]
   where $q_{e,p}$ is $y_{a,p}$ if $e=a \in A$, $z_{b,p}$ if $e=b \in B$.

4. **Non-negativity:**
   \[
   x_p \geq 0, \quad y_{a,p} \geq 0, \quad z_{b,p} \geq 0 \quad \forall p, a, b
   \]

#### Data Mapping

- Table: 43.csv (table_id: file_0_view_0)
    - Product list: columns "Product I", "Product II", "Product III"
    - Equipment list: column "Equipment / Cost"
    - Processing times: intersection of equipment rows and product columns
    - Raw material cost: row where "Equipment / Cost" = "Raw Material Cost (yuan/unit)"
    - Selling price: row where "Equipment / Cost" = "Unit Price (yuan/unit)"
    - Available equipment operating time: column "Available Equipment Operating Time"
    - Equipment cost at full load: column "Equipment Cost at Full Load (yuan)"
    - Assignment constraints: encoded by presence/absence of processing time values in product columns for each equipment row

- Assignment sets $\delta_{a,p}$ and $\delta_{b,p}$ are determined by non-empty entries in the corresponding cells.

---

This model maximizes total profit (sales minus raw material and equipment costs), subject to production assignment, equipment time, and assignment feasibility constraints, using all data fields as mapped above.