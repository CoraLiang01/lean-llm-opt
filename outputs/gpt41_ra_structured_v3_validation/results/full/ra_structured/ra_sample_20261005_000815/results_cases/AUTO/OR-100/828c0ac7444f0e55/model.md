Let:
- \( x_i \) = number of units to produce of component \( i \) (\( i \in \{C1, C2, ..., C111\} \)), where \( x_i \in \mathbb{Z}_{\geq 0} \).
- \( p_i \) = unit price of component \( i \) (from unit_price.csv).
- \( a_{wi} \) = unit time required for component \( i \) in workshop \( w \) (from processing_time_unit.csv, for \( w \) in {Casting, Milling, Finishing, Assembly, QA & Packaging}).
- \( b_w \) = total available working hours in workshop \( w \) (from total_working_hours.csv).

**Objective:**
\[
\max \sum_{i \in \{C1,\ldots,C111\}} p_i x_i
\]

**Subject to (for each workshop \( w \)):**
\[
\sum_{i \in \{C1,\ldots,C111\}} a_{wi} x_i \leq b_w
\]
where:
- For Casting: \( b_{\text{Casting}} = 7650 \)
- For Milling: \( b_{\text{Milling}} = 6320 \)
- For Finishing: \( b_{\text{Finishing}} = 5538 \)
- For Assembly: \( b_{\text{Assembly}} = 5957 \)
- For QA & Packaging: \( b_{\text{QA \& Packaging}} = 6988 \)

**Variable domains:**
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{C1,\ldots,C111\}
\]

---

### Explicit Formulation

Let the set of components be \( I = \{C1, C2, ..., C111\} \).

Let the set of workshops be \( W = \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\} \).

**Parameters:**
- \( p_i \): unit price of component \( i \) (from unit_price.csv)
- \( a_{wi} \): unit time required for component \( i \) in workshop \( w \) (from processing_time_unit.csv)
- \( b_w \): total available working hours in workshop \( w \) (from total_working_hours.csv)

**Decision variables:**
- \( x_i \): number of units to produce of component \( i \), integer, \( \geq 0 \)

**Model:**

\[
\begin{align*}
\max \quad & \sum_{i \in I} p_i x_i \\
\text{s.t.} \quad & \sum_{i \in I} a_{wi} x_i \leq b_w, \quad \forall w \in W \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]

**Where:**

- \( p_i \) is given by the "unit_price" column in unit_price.csv, matched by "Unnamed: 0" (component ID).
- \( a_{wi} \) is given by the value in row \( w \), column \( i \) of processing_time_unit.csv.
- \( b_w \) is given by the "total_hours" column in total_working_hours.csv, matched by "workshop".

**All data must be used as returned, preserving the original order and identifiers.**