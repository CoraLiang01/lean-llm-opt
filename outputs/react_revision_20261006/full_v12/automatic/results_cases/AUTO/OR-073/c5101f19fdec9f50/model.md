## Symbolic Mathematical Model

**Sets**
- $P = \{\text{I}, \text{II}, \text{III}\}$: Products
- $A = \{\text{A1}, \text{A2}\}$: Equipment for procedure A
- $B = \{\text{B1}, \text{B2}, \text{B3}\}$: Equipment for procedure B

**Parameters** (from file_0_view_0)
- $t_{a,p}$: Processing time per unit of product $p$ on equipment $a \in A$ (minutes/unit)
- $t_{b,p}$: Processing time per unit of product $p$ on equipment $b \in B$ (minutes/unit)
- $T_a$: Available operating time for equipment $a$ (minutes)
- $T_b$: Available operating time for equipment $b$ (minutes)
- $C_a$: Equipment cost at full load for $a$ (yuan)
- $C_b$: Equipment cost at full load for $b$ (yuan)
- $c_p$: Raw material cost per unit of product $p$ (yuan/unit)
- $s_p$: Selling price per unit of product $p$ (yuan/unit)

**Decision Variables**
- $x_{a,p} \geq 0$: Quantity of product $p$ processed on equipment $a \in A$
- $y_{b,p} \geq 0$: Quantity of product $p$ processed on equipment $b \in B$

**Objective**
Maximize total profit:
\[
\max \sum_{p \in P} s_p \cdot q_p - \sum_{p \in P} c_p \cdot q_p - \sum_{a \in A} \frac{C_a}{T_a} \cdot \sum_{p \in P_a} t_{a,p} x_{a,p} - \sum_{b \in B} \frac{C_b}{T_b} \cdot \sum_{p \in P_b} t_{b,p} y_{b,p}
\]
where $q_p$ is the total production of product $p$:
\[
q_{\text{I}} = \sum_{a \in A} x_{a,\text{I}} = \sum_{b \in B} y_{b,\text{I}}
\]
\[
q_{\text{II}} = \sum_{a \in A} x_{a,\text{II}} = y_{\text{B1},\text{II}}
\]
\[
q_{\text{III}} = x_{\text{A2},\text{III}} = y_{\text{B2},\text{III}}
\]

**Constraints**

1. **Equipment time limits**
   - For all $a \in A$:
     \[
     \sum_{p \in P_a} t_{a,p} x_{a,p} \leq T_a
     \]
   - For all $b \in B$:
     \[
     \sum_{p \in P_b} t_{b,p} y_{b,p} \leq T_b
     \]

2. **Product routing and eligibility**
   - $x_{a,p} = 0$ if product $p$ cannot be processed on $a$ (see eligibility below)
   - $y_{b,p} = 0$ if product $p$ cannot be processed on $b$ (see eligibility below)

   Eligibility (from question and data):
   - Product I: $x_{a,\text{I}}$ for $a \in A$; $y_{b,\text{I}}$ for $b \in B$
   - Product II: $x_{a,\text{II}}$ for $a \in A$; $y_{\text{B1},\text{II}}$ only
   - Product III: $x_{\text{A2},\text{III}}$ only; $y_{\text{B2},\text{III}}$ only

3. **Flow conservation (each product's output after A equals output after B)**
   - Product I:
     \[
     \sum_{a \in A} x_{a,\text{I}} = \sum_{b \in B} y_{b,\text{I}}
     \]
   - Product II:
     \[
     \sum_{a \in A} x_{a,\text{II}} = y_{\text{B1},\text{II}}
     \]
   - Product III:
     \[
     x_{\text{A2},\text{III}} = y_{\text{B2},\text{III}}
     \]

4. **Non-negativity**
   \[
   x_{a,p} \geq 0,\quad y_{b,p} \geq 0
   \]
   for all eligible $(a,p)$ and $(b,p)$.

---

### Data Mapping

- **file_0_view_0**:
  - Equipment: rows with "A1", "A2" $\to$ $A$; "B1", "B2", "B3" $\to$ $B$
  - $t_{a,p}$: value in column "Product $p$" for row "Equipment / Cost" = $a$
  - $t_{b,p}$: value in column "Product $p$" for row "Equipment / Cost" = $b$
  - $T_a$, $T_b$: "Available Equipment Operating Time" for $a$, $b$
  - $C_a$, $C_b$: "Equipment Cost at Full Load (yuan)" for $a$, $b$
  - $c_p$: row "Raw Material Cost (yuan/unit)", column "Product $p$"
  - $s_p$: row "Unit Price (yuan/unit)", column "Product $p$"

- **Eligibility**:
  - $x_{a,p}$ allowed if $t_{a,p}$ is not blank
  - $y_{b,p}$ allowed if $t_{b,p}$ is not blank

---

**All indices, parameters, and constraints are mapped directly from file_0_view_0 as described above.**