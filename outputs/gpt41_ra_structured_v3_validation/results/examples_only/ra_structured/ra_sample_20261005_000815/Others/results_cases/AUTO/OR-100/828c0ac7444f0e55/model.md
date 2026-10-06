Let:
- $I$ = set of component types, indexed by $i$ (from "C1" to "C111")
- $W$ = set of workshops, indexed by $w$ ("Casting", "Milling", "Finishing", "Assembly", "QA \& Packaging")
- $x_i$ = number of units of component $i$ to produce (decision variable, nonnegative integer)

Parameters:
- $p_i$ = unit price of component $i$ (from unit_price.csv)
- $a_{w,i}$ = unit time required for component $i$ in workshop $w$ (from processing_time_unit.csv)
- $c_w$ = total available working hours in workshop $w$ (from total_working_hours.csv)

**Mathematical Model**

**Variables:**
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Subject to:**

For each workshop $w \in W$:
\[
\sum_{i \in I} a_{w,i} x_i \leq c_w
\]

**Explicitly, using the retrieved data:**

Let $I = \{\text{C1}, \text{C2}, \ldots, \text{C111}\}$

Let $W = \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$

Let $p_i$ be as in unit_price.csv, e.g., $p_{\text{C1}} = 193$, $p_{\text{C2}} = 64$, ..., $p_{\text{C111}} = 142$.

Let $a_{w,i}$ be as in processing_time_unit.csv, e.g., $a_{\text{Casting},\text{C1}} = 0.74$, $a_{\text{Milling},\text{C1}} = 0.6$, ..., $a_{\text{QA \& Packaging},\text{C111}} = 3.21$.

Let $c_{\text{Casting}} = 7650$, $c_{\text{Milling}} = 6320$, $c_{\text{Finishing}} = 5538$, $c_{\text{Assembly}} = 5957$, $c_{\text{QA \& Packaging}} = 6988$.

**Full Model:**

\[
\begin{align*}
\max \quad & \sum_{i \in I} p_i x_i \\
\text{s.t.} \quad
& \sum_{i \in I} a_{\text{Casting},i} x_i \leq 7650 \\
& \sum_{i \in I} a_{\text{Milling},i} x_i \leq 6320 \\
& \sum_{i \in I} a_{\text{Finishing},i} x_i \leq 5538 \\
& \sum_{i \in I} a_{\text{Assembly},i} x_i \leq 5957 \\
& \sum_{i \in I} a_{\text{QA \& Packaging},i} x_i \leq 6988 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\end{align*}
\]

**Where:**
- $p_i$ is the unit price of component $i$ (from unit_price.csv)
- $a_{w,i}$ is the unit time required for component $i$ in workshop $w$ (from processing_time_unit.csv)
- $c_w$ is the total available working hours in workshop $w$ (from total_working_hours.csv)
- $x_i$ is the number of units of component $i$ to produce (nonnegative integer)

**All identifiers and coefficients are to be used exactly as retrieved from the CSVs.**