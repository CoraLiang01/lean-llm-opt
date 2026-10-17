Let:
- $S$ = set of source locations = $\{\text{S1}, \text{S2}, ..., \text{S10}\}$
- $D$ = set of demand locations = $\{\text{D1}, \text{D2}, ..., \text{D20}\}$
- $c_{sd}$ = unit transportation cost from source $s$ to destination $d$ (from expanded_cost_matrix.csv)
- $supply_s$ = supply at source $s$ (from expanded_sources.csv)
- $demand_d$ = demand at destination $d$ (from expanded_destinations.csv)

Decision variables:
- $x_{sd} \geq 0$: amount of cargo shipped from source $s$ to destination $d$ (continuous, can be fractional)
- $t_{sd} \in \mathbb{Z}_{\geq 0}$: number of trucks dispatched from $s$ to $d$ (integer, whole trucks only)

Parameters:

From expanded_sources.csv:
- S1: $supply_{S1} = 103$
- S2: $supply_{S2} = 87$
- S3: $supply_{S3} = 95$
- S4: $supply_{S4} = 112$
- S5: $supply_{S5} = 97$
- S6: $supply_{S6} = 103$
- S7: $supply_{S7} = 101$
- S8: $supply_{S8} = 94$
- S9: $supply_{S9} = 102$
- S10: $supply_{S10} = 106$

From expanded_destinations.csv:
- D1: $demand_{D1} = 61$
- D2: $demand_{D2} = 54$
- D3: $demand_{D3} = 56$
- D4: $demand_{D4} = 54$
- D5: $demand_{D5} = 53$
- D6: $demand_{D6} = 47$
- D7: $demand_{D7} = 56$
- D8: $demand_{D8} = 57$
- D9: $demand_{D9} = 56$
- D10: $demand_{D10} = 34$
- D11: $demand_{D11} = 55$
- D12: $demand_{D12} = 53$
- D13: $demand_{D13} = 37$
- D14: $demand_{D14} = 31$
- D15: $demand_{D15} = 62$
- D16: $demand_{D16} = 58$
- D17: $demand_{D17} = 39$
- D18: $demand_{D18} = 32$
- D19: $demand_{D19} = 38$
- D20: $demand_{D20} = 67$

From expanded_cost_matrix.csv (partial, see full data above for all $c_{sd}$):

For example, for S1:
- $c_{S1,D1} = 2.75$, $c_{S1,D2} = 2.6$, ..., $c_{S1,D20} = 10.91$
(similarly for all $s \in S$, $d \in D$)

Model:

Minimize total transportation cost:
$$
\min \sum_{s \in S} \sum_{d \in D} c_{sd} \cdot x_{sd}
$$

Subject to:

1. Supply constraints (at each source):
$$
\sum_{d \in D} x_{sd} \leq supply_s \qquad \forall s \in S
$$

2. Demand constraints (at each destination):
$$
\sum_{s \in S} x_{sd} = demand_d \qquad \forall d \in D
$$

3. Truck loading constraints (partial loads allowed, but only whole trucks dispatched):
$$
x_{sd} \leq 10 \cdot t_{sd} \qquad \forall s \in S, d \in D
$$

4. Variable domains:
$$
x_{sd} \geq 0 \qquad \forall s \in S, d \in D \\
t_{sd} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S, d \in D
$$

All parameters and indices are as retrieved above, with no omitted data.

Summary of retrieved data:

- Sources and supplies: S1 (103), S2 (87), S3 (95), S4 (112), S5 (97), S6 (103), S7 (101), S8 (94), S9 (102), S10 (106)
- Destinations and demands: D1 (61), D2 (54), ..., D20 (67)
- Unit costs $c_{sd}$: as in expanded_cost_matrix.csv for all $s \in S$, $d \in D$

This is a complete mixed-integer linear programming model for the described transportation problem with truck integer constraints and partial loading.