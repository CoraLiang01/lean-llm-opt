##### Mathematical Model

Let:
- $P = \{\text{I}, \text{II}, \text{III}\}$: set of products
- $A = \{\text{A1}, \text{A2}\}$: set of equipment for procedure A
- $B = \{\text{B1}, \text{B2}, \text{B3}\}$: set of equipment for procedure B

Decision variables:
- $x_p \geq 0$: quantity of product $p \in P$ produced (continuous)
- $y_{e,p}^A \geq 0$: amount of product $p$ processed on equipment $e \in A$ for procedure A
- $y_{e,p}^B \geq 0$: amount of product $p$ processed on equipment $e \in B$ for procedure B

Parameters (from Data Mapping below):
- $t_{e,p}^A$: processing time per unit of product $p$ on equipment $e$ for procedure A
- $t_{e,p}^B$: processing time per unit of product $p$ on equipment $e$ for procedure B
- $T_e$: available operating time for equipment $e$
- $C_e$: equipment cost at full load for equipment $e$
- $c_p^{\text{raw}}$: raw material cost per unit of product $p$
- $r_p$: selling price per unit of product $p$

Objective:
\[
\max \sum_{p \in P} \left[ r_p x_p - c_p^{\text{raw}} x_p \right] - \sum_{e \in A \cup B} C_e \cdot \frac{1}{T_e} \sum_{p} t_{e,p} y_{e,p}
\]
where $t_{e,p}$ and $y_{e,p}$ are defined for each procedure as appropriate.

Subject to:

1. **Production-Processing Consistency:**
   - For each $p \in P$:
     \[
     x_p = \sum_{e \in A_p} y_{e,p}^A = \sum_{e \in B_p} y_{e,p}^B
     \]
     where $A_p$ and $B_p$ are the sets of eligible equipment for product $p$ in procedures A and B, respectively.

2. **Equipment Time Constraints:**
   - For each $e \in A$:
     \[
     \sum_{p \in P_e^A} t_{e,p}^A y_{e,p}^A \leq T_e
     \]
   - For each $e \in B$:
     \[
     \sum_{p \in P_e^B} t_{e,p}^B y_{e,p}^B \leq T_e
     \]
     where $P_e^A$ (resp. $P_e^B$) is the set of products that can be processed on equipment $e$ for procedure A (resp. B).

3. **Eligibility Constraints:**
   - $y_{e,p}^A = 0$ if product $p$ cannot be processed on equipment $e$ for procedure A.
   - $y_{e,p}^B = 0$ if product $p$ cannot be processed on equipment $e$ for procedure B.

4. **Nonnegativity:**
   - $x_p \geq 0$, $y_{e,p}^A \geq 0$, $y_{e,p}^B \geq 0$ for all defined indices.

##### Data Mapping

- Table: file_0_view_0 (43.csv)
    - Equipment / Cost: equipment label ($e$)
    - Product I, Product II, Product III: processing time per unit ($t_{e,p}$) for each product $p$ on equipment $e$
    - Available Equipment Operating Time: $T_e$ for equipment $e$
    - Equipment Cost at Full Load (yuan): $C_e$ for equipment $e$
    - Row with Equipment / Cost = "Raw Material Cost (yuan/unit)": $c_p^{\text{raw}}$ for each product $p$
    - Row with Equipment / Cost = "Unit Price (yuan/unit)": $r_p$ for each product $p$

- Eligibility sets:
    - For procedure A:
        - Product I: $A_p = \{\text{A1}, \text{A2}\}$
        - Product II: $A_p = \{\text{A1}, \text{A2}\}$
        - Product III: $A_p = \{\text{A2}\}$
    - For procedure B:
        - Product I: $B_p = \{\text{B1}, \text{B2}, \text{B3}\}$
        - Product II: $B_p = \{\text{B1}\}$
        - Product III: $B_p = \{\text{B2}\}$

- $P_e^A$ and $P_e^B$ are the sets of products that can be processed on equipment $e$ for each procedure, as above.

- $t_{e,p}^A$ and $t_{e,p}^B$ are taken from the corresponding cells for each equipment and product in the table, for A and B equipment respectively.

- $T_e$ and $C_e$ are taken from the corresponding columns for each equipment.

- $c_p^{\text{raw}}$ and $r_p$ are taken from the rows labeled "Raw Material Cost (yuan/unit)" and "Unit Price (yuan/unit)", respectively.

**All index sets, parameters, and eligibility are mapped directly from file_0_view_0 (43.csv) as described above.**