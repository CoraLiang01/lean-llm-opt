#### Abstract Optimization Model

**Index Sets:**
- $P$: Set of products (from columns "Product I", "Product II", "Product III" in file_0_view_0)
- $A$: Set of equipment for procedure A (A1, A2)
- $B$: Set of equipment for procedure B (B1, B2, B3)
- $E$: Set of all equipment (A1, A2, B1, B2, B3)

**Parameters:**
- $t_{e,p}$: Processing time required on equipment $e \in E$ for product $p \in P$ (from file_0_view_0, columns "Product I", "Product II", "Product III", rows for each equipment)
- $T_e$: Available operating time for equipment $e \in E$ (from "Available Equipment Operating Time", file_0_view_0)
- $C_e$: Equipment cost at full load for equipment $e \in E$ (from "Equipment Cost at Full Load (yuan)", file_0_view_0)
- $c_p$: Raw material cost per unit of product $p$ (from row "Raw Material Cost (yuan/unit)", file_0_view_0)
- $s_p$: Selling price per unit of product $p$ (from row "Unit Price (yuan/unit)", file_0_view_0)

**Decision Variables:**
- $x_p \geq 0$: Production quantity of product $p \in P$ (continuous)
- $y_{e,p} \geq 0$: Amount of product $p$ processed on equipment $e$ (continuous)

**Objective:**
\[
\max \left[ \sum_{p \in P} s_p x_p - \sum_{p \in P} c_p x_p - \sum_{e \in E} C_e \cdot \frac{1}{T_e} \sum_{p \in P} t_{e,p} y_{e,p} \right]
\]

**Constraints:**

1. **Production Assignment:**
   - For each product $p$ and each procedure (A and B), assign production to eligible equipment:
     - For procedure A:
       - Product I: $y_{A1,I} + y_{A2,I} = x_I$
       - Product II: $y_{A1,II} + y_{A2,II} = x_{II}$
       - Product III: $y_{A2,III} = x_{III}$
     - For procedure B:
       - Product I: $y_{B1,I} + y_{B2,I} + y_{B3,I} = x_I$
       - Product II: $y_{B1,II} = x_{II}$
       - Product III: $y_{B2,III} = x_{III}$

2. **Equipment Capacity:**
   - For each equipment $e \in E$:
     \[
     \sum_{p \in P} t_{e,p} y_{e,p} \leq T_e
     \]
     (where $t_{e,p}$ is defined only if equipment $e$ can process product $p$; otherwise, $y_{e,p} = 0$)

3. **Eligibility Constraints:**
   - $y_{e,p} = 0$ if equipment $e$ cannot process product $p$ (as per the question's eligibility rules and data in file_0_view_0).

4. **Non-negativity:**
   - $x_p \geq 0$, $y_{e,p} \geq 0$ for all $p \in P$, $e \in E$

---

#### Data Mapping

- **file_0_view_0, column "Equipment / Cost"**: Identifies equipment and cost rows.
- **file_0_view_0, columns "Product I", "Product II", "Product III"**: Processing times per equipment and product; raw material cost and unit price in special rows.
- **file_0_view_0, column "Available Equipment Operating Time"**: $T_e$ for each equipment.
- **file_0_view_0, column "Equipment Cost at Full Load (yuan)"**: $C_e$ for each equipment.
- **file_0_view_0, row "Raw Material Cost (yuan/unit)"**: $c_p$ for each product.
- **file_0_view_0, row "Unit Price (yuan/unit)"**: $s_p$ for each product.

Returned data is the full table from 43.csv, as required by the query. No additional filtering was applied. All index sets, parameters, and eligibility rules are defined using the returned data and the user’s description.