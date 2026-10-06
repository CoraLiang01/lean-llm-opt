**Abstract Mathematical Model**

---

**Index Sets:**

- $P$: Set of products, $P = \{\text{I}, \text{II}, \text{III}\}$
- $E_A$: Set of equipment for procedure A, $E_A = \{\text{A1}, \text{A2}\}$
- $E_B$: Set of equipment for procedure B, $E_B = \{\text{B1}, \text{B2}, \text{B3}\}$

---

**Parameters:**

- $t_{e,p}$: Processing time (hours per unit) for product $p$ on equipment $e$ (from "Product I", "Product II", "Product III" columns for each equipment in 43.csv)
- $T_e$: Available operating time (hours) for equipment $e$ ("Available Equipment Operating Time" in 43.csv)
- $C_e$: Equipment cost at full load (yuan) for equipment $e$ ("Equipment Cost at Full Load (yuan)" in 43.csv)
- $c_p^{\text{raw}}$: Raw material cost per unit for product $p$ (row "Raw Material Cost (yuan/unit)" in 43.csv)
- $r_p$: Selling price per unit for product $p$ (row "Unit Price (yuan/unit)" in 43.csv)

---

**Decision Variables:**

- $x_{A,e,p} \geq 0$: Number of units of product $p$ processed on equipment $e$ for procedure A (continuous)
- $x_{B,e,p} \geq 0$: Number of units of product $p$ processed on equipment $e$ for procedure B (continuous)

---

**Objective Function:**

\[
\max \left\{ \sum_{p \in P} r_p \cdot y_p - \sum_{p \in P} c_p^{\text{raw}} \cdot y_p - \sum_{e} C_e \cdot \frac{u_e}{T_e} \right\}
\]

where:
- $y_p$: Total units of product $p$ produced (must be the same through both procedures)
- $u_e$: Total time used on equipment $e$ (see below)

---

**Constraints:**

1. **Production Consistency:**
   - For each product $p$:
     \[
     y_p = \sum_{e \in E_A(p)} x_{A,e,p} = \sum_{e \in E_B(p)} x_{B,e,p}
     \]
     where $E_A(p)$ and $E_B(p)$ are the sets of eligible equipment for product $p$ in procedures A and B, respectively (see eligibility below).

2. **Equipment Time Capacity:**
   - For each equipment $e$:
     \[
     \sum_{p \in P(e)} t_{e,p} \cdot x_{A,e,p} + \sum_{p \in P(e)} t_{e,p} \cdot x_{B,e,p} \leq T_e
     \]
     where $P(e)$ is the set of products that can be processed on equipment $e$ for the relevant procedure.

     (For A equipment, only $x_{A,e,p}$ terms are nonzero; for B equipment, only $x_{B,e,p}$ terms are nonzero.)

3. **Eligibility Constraints:**
   - $x_{A,e,p} = 0$ if $e \notin E_A(p)$
   - $x_{B,e,p} = 0$ if $e \notin E_B(p)$

   Where:
   - $E_A(\text{I}) = \{\text{A1}, \text{A2}\}$, $E_B(\text{I}) = \{\text{B1}, \text{B2}, \text{B3}\}$
   - $E_A(\text{II}) = \{\text{A1}, \text{A2}\}$, $E_B(\text{II}) = \{\text{B1}\}$
   - $E_A(\text{III}) = \{\text{A2}\}$, $E_B(\text{III}) = \{\text{B2}\}$

4. **Equipment Cost Calculation:**
   - For each equipment $e$:
     \[
     u_e = \sum_{p \in P(e)} t_{e,p} \cdot x_{A,e,p} + \sum_{p \in P(e)} t_{e,p} \cdot x_{B,e,p}
     \]
     (Again, only one of the two sums is nonzero for each $e$.)

---

**Variable Domains:**

- $x_{A,e,p} \geq 0$, continuous
- $x_{B,e,p} \geq 0$, continuous
- $y_p \geq 0$, continuous

---

**Data Mapping**

- **Index Sets:**
  - $P$: Products = {"Product I", "Product II", "Product III"} (from columns in 43.csv)
  - $E_A$: Equipment / Cost = {"A1", "A2"} (from "Equipment / Cost" in 43.csv, rows with A1, A2)
  - $E_B$: Equipment / Cost = {"B1", "B2", "B3"} (from "Equipment / Cost" in 43.csv, rows with B1, B2, B3)

- **Parameters:**
  - $t_{e,p}$: file_0_view_0, columns "Product I", "Product II", "Product III", rows with "Equipment / Cost" = $e$
  - $T_e$: file_0_view_0, column "Available Equipment Operating Time", row with "Equipment / Cost" = $e$
  - $C_e$: file_0_view_0, column "Equipment Cost at Full Load (yuan)", row with "Equipment / Cost" = $e$
  - $c_p^{\text{raw}}$: file_0_view_0, row with "Equipment / Cost" = "Raw Material Cost (yuan/unit)", columns "Product I", "Product II", "Product III"
  - $r_p$: file_0_view_0, row with "Equipment / Cost" = "Unit Price (yuan/unit)", columns "Product I", "Product II", "Product III"

---

**Notes:**

- Only allow $x_{A,e,p}$ and $x_{B,e,p}$ for eligible $(e,p)$ pairs as described in the query.
- All variables are continuous and nonnegative.
- Equipment cost is charged proportionally to the fraction of time used ($u_e/T_e$) at full load cost $C_e$.

---

**End of Model**