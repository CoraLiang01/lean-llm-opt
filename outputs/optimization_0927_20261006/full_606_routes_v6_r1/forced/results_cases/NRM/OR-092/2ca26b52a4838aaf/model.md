#### Abstract Model

**Index Sets:**
- $S$: set of source locations (from expanded_sources.csv, column source_id)
- $D$: set of demand locations (from expanded_destinations.csv, column destination_id)

**Parameters:**
- $a_s$: supply units available at source $s \in S$ (from expanded_sources.csv, column supply_units)
- $b_d$: demand units required at destination $d \in D$ (from expanded_destinations.csv, column demand_units)
- $c_{sd}$: unit transportation cost from source $s \in S$ to destination $d \in D$ (from expanded_cost_matrix.csv, entry at row source_id $s$, column $d$)
- $Q$: truck capacity (fixed at 10 units per truck)

**Decision Variables:**
- $y_{sd} \in \mathbb{Z}_+$: number of trucks dispatched from source $s$ to destination $d$
- $x_{sd} \in [0, Q]$: units of cargo shipped from source $s$ to destination $d$

**Objective:**
\[
\min \sum_{s \in S} \sum_{d \in D} c_{sd} \cdot x_{sd}
\]

**Constraints:**

1. **Truck Loading and Integer Trips:**
   \[
   x_{sd} \leq Q \cdot y_{sd} \qquad \forall s \in S,\, d \in D
   \]
   \[
   y_{sd} \in \mathbb{Z}_+, \quad x_{sd} \geq 0 \qquad \forall s \in S,\, d \in D
   \]

2. **Supply Constraints:**
   \[
   \sum_{d \in D} x_{sd} \leq a_s \qquad \forall s \in S
   \]

3. **Demand Constraints:**
   \[
   \sum_{s \in S} x_{sd} = b_d \qquad \forall d \in D
   \]

4. **Truck Capacity:**
   \[
   0 \leq x_{sd} \leq Q \cdot y_{sd} \leq Q \cdot U_{sd} \qquad \forall s \in S,\, d \in D
   \]
   (where $U_{sd}$ is an upper bound on the number of trucks, e.g., $U_{sd} = \lceil \min(a_s, b_d)/Q \rceil$; this can be omitted if not needed for implementation.)

**Variable Domains:**
- $y_{sd} \in \mathbb{Z}_+$ (non-negative integers)
- $x_{sd} \in [0, Q \cdot y_{sd}]$ (continuous, non-negative, upper-bounded by truck count times capacity)

---

#### Data Mapping

- **expanded_sources.csv**: 
  - Table ID: file_2_view_0
  - Columns: source_id $\rightarrow S$, supply_units $\rightarrow a_s$
- **expanded_destinations.csv**: 
  - Table ID: file_1_view_0
  - Columns: destination_id $\rightarrow D$, demand_units $\rightarrow b_d$
- **expanded_cost_matrix.csv**: 
  - Table ID: file_0_view_0
  - Row index: source_id $\rightarrow S$
  - Column index: destination_id $\rightarrow D$
  - Entries: $c_{sd}$

---

**Notes:**
- All supply, demand, and cost data are mapped directly from the returned records.
- The model ensures all demand is met, supply is not exceeded, and truck dispatches are integer, with partial loading allowed per truck.
- Costs are per unit shipped, not per truck.