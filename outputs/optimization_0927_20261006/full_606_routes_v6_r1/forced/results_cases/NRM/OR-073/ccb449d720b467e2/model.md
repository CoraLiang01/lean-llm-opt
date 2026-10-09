#### Index Sets

- $P$: Set of products, $P = \{\text{I}, \text{II}, \text{III}\}$
- $E_A$: Set of equipment for procedure A, $E_A = \{\text{A1}, \text{A2}\}$
- $E_B$: Set of equipment for procedure B, $E_B = \{\text{B1}, \text{B2}, \text{B3}\}$

#### Parameters

- $t_{e,p}$: Processing time (hours/unit) for product $p \in P$ on equipment $e$ (from "Product I", "Product II", "Product III" columns for each equipment row)
- $T_e$: Available operating time (hours) for equipment $e$
- $C_e$: Equipment cost at full load (yuan) for equipment $e$
- $c_p$: Raw material cost per unit for product $p$
- $s_p$: Selling price per unit for product $p$

- Eligibility sets:
    - For procedure A:
        - Product I: $E_A^{\text{I}} = \{\text{A1}, \text{A2}\}$
        - Product II: $E_A^{\text{II}} = \{\text{A1}, \text{A2}\}$
        - Product III: $E_A^{\text{III}} = \{\text{A2}\}$
    - For procedure B:
        - Product I: $E_B^{\text{I}} = \{\text{B1}, \text{B2}, \text{B3}\}$
        - Product II: $E_B^{\text{II}} = \{\text{B1}\}$
        - Product III: $E_B^{\text{III}} = \{\text{B2}\}$

#### Decision Variables

- $x_p \geq 0$: Number of units produced of product $p \in P$ (continuous)
- $y_{e,p} \geq 0$: Number of units of product $p$ processed on equipment $e$ for procedure A, $e \in E_A^{p}$
- $z_{e,p} \geq 0$: Number of units of product $p$ processed on equipment $e$ for procedure B, $e \in E_B^{p}$

#### Objective Function

Maximize total profit:
\[
\max \left[ \sum_{p \in P} s_p x_p - \sum_{p \in P} c_p x_p - \sum_{e \in E_A \cup E_B} C_e \cdot \frac{1}{T_e} \cdot \left( \sum_{p: e \in E_A^p} t_{e,p} y_{e,p} + \sum_{p: e \in E_B^p} t_{e,p} z_{e,p} \right) \right]
\]

#### Constraints

1. **Production Assignment (Procedure A):**
   \[
   \sum_{e \in E_A^p} y_{e,p} = x_p, \quad \forall p \in P
   \]

2. **Production Assignment (Procedure B):**
   \[
   \sum_{e \in E_B^p} z_{e,p} = x_p, \quad \forall p \in P
   \]

3. **Equipment Time Limits:**
   \[
   \sum_{p: e \in E_A^p} t_{e,p} y_{e,p} + \sum_{p: e \in E_B^p} t_{e,p} z_{e,p} \leq T_e, \quad \forall e \in E_A \cup E_B
   \]

4. **Non-negativity:**
   \[
   x_p \geq 0, \quad y_{e,p} \geq 0, \quad z_{e,p} \geq 0, \quad \forall p, e
   \]

#### Data Mapping

- Table: 43.csv (table_id: file_0_view_0)
    - "Equipment / Cost": Equipment identifiers (A1, A2, B1, B2, B3)
    - "Product I", "Product II", "Product III": Processing time (hours/unit) for each product on each equipment
    - "Available Equipment Operating Time": $T_e$ for each equipment
    - "Equipment Cost at Full Load (yuan)": $C_e$ for each equipment
    - Row with "Raw Material Cost (yuan/unit)": $c_p$ for each product
    - Row with "Unit Price (yuan/unit)": $s_p$ for each product

Eligibility for each product-equipment pair is determined by the presence of a value in the corresponding cell; empty cells indicate ineligibility. Only equipment and products as described in the user query are included.