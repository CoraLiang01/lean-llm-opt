### Mathematical Model

**Index Sets:**
- $I$: set of components (from `unit_price.csv`, column `Unnamed: 0`)
- $W$: set of workshops (from `processing_time_unit.csv` row `Unnamed: 0` and `total_working_hours.csv` column `workshop`)

**Parameters:**
- $p_i$: unit price of component $i$ (`unit_price.csv`, column `unit_price`)
- $a_{wi}$: unit processing time of component $i$ in workshop $w$ (`processing_time_unit.csv`, row `Unnamed: 0` for $w$, column $i$)
- $h_w$: total available working hours in workshop $w$ (`total_working_hours.csv`, column `total_hours`)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of component $i$ to produce

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Constraints:**
\[
\sum_{i \in I} a_{wi} x_i \leq h_w \qquad \forall w \in W
\]
\[
x_i \geq 0 \text{ and integer} \qquad \forall i \in I
\]

---

### Data Mapping

- $I$: All values in `unit_price.csv`, column `Unnamed: 0`
- $W$: All values in `processing_time_unit.csv`, row `Unnamed: 0` and `total_working_hours.csv`, column `workshop`
- $p_i$: `unit_price.csv`, column `unit_price`, keyed by `Unnamed: 0`
- $a_{wi}$: `processing_time_unit.csv`, row `Unnamed: 0` for $w$, column $i$
- $h_w$: `total_working_hours.csv`, column `total_hours`, keyed by `workshop`
- $x_i$: Decision variable for each $i \in I$ (component)

---

**All index sets, parameters, and constraints are mapped directly from the current CSV data as described above.**