#### Mathematical Model

Let:
- $I$ = set of component types (from unit_price.csv, column Unnamed: 0)
- $K$ = set of workshops (from processing_time_unit.csv, column Unnamed: 0 and total_working_hours.csv, column workshop)
- $p_i$ = unit price of component $i$ (from unit_price.csv, column unit_price)
- $a_{ki}$ = unit processing time of component $i$ in workshop $k$ (from processing_time_unit.csv, row $k$, column $i$)
- $b_k$ = total available working hours in workshop $k$ (from total_working_hours.csv, column total_hours)
- $x_i$ = number of units of component $i$ to produce (decision variable, nonnegative integer)

Objective:
\[
\max \sum_{i \in I} p_i x_i
\]

Subject to:
\[
\sum_{i \in I} a_{ki} x_i \leq b_k \qquad \forall k \in K
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
\]

---

#### Data Mapping

- $I$: All values in unit_price.csv, column Unnamed: 0
- $K$: All values in processing_time_unit.csv, column Unnamed: 0 (workshop names), matched to total_working_hours.csv, column workshop
- $p_i$: unit_price.csv, column unit_price, key Unnamed: 0
- $a_{ki}$: processing_time_unit.csv, row Unnamed: 0 = $k$, column $i$
- $b_k$: total_working_hours.csv, column total_hours, key workshop
- $x_i$: Decision variable for each $i \in I$ (component type)

All indices, parameters, and constraints are mapped directly to the supplied CSV columns and business identifiers.