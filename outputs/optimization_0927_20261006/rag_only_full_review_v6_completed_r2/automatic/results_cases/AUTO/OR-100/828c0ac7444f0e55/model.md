Let:
- \( I = \{C1, C2, ..., C111\} \) be the set of component types (component IDs as in the data).
- \( W = \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\} \) be the set of workshops.
- For each component \( i \in I \), let \( x_i \) be the number of units of component \( i \) to produce (decision variable, integer, \( x_i \geq 0 \)).
- For each component \( i \), let \( p_i \) be its unit price (from unit_price.csv).
- For each workshop \( w \in W \), let \( T_w \) be the total available working hours (from total_working_hours.csv).
- For each workshop \( w \) and component \( i \), let \( a_{w,i} \) be the unit processing time required for component \( i \) in workshop \( w \) (from processing_time_unit.csv).

The model is:

Variables:
- \( x_i \in \mathbb{Z}_+, \forall i \in I \) (nonnegative integers)

Objective:
\[
\max \sum_{i \in I} p_i x_i
\]
where \( p_i \) is the unit price of component \( i \).

Constraints:
For each workshop \( w \in W \):
\[
\sum_{i \in I} a_{w,i} x_i \leq T_w
\]
where:
- \( a_{w,i} \) is the processing time per unit of component \( i \) in workshop \( w \) (from the corresponding row and column in processing_time_unit.csv).
- \( T_w \) is the total available working hours for workshop \( w \) (from total_working_hours.csv).

Explicitly, using the supplied data:

Let the workshops and their total available hours be:
- Casting: \( T_{\text{Casting}} = 7650 \)
- Milling: \( T_{\text{Milling}} = 6320 \)
- Finishing: \( T_{\text{Finishing}} = 5538 \)
- Assembly: \( T_{\text{Assembly}} = 5957 \)
- QA & Packaging: \( T_{\text{QA \& Packaging}} = 6988 \)

For each component \( i \in \{C1, ..., C111\} \), let \( x_i \in \mathbb{Z}_+ \).

Objective:
\[
\max \sum_{i = 1}^{111} \text{unit\_price}_i \cdot x_{C_i}
\]
where \(\text{unit\_price}_i\) is as in unit_price.csv for component \( C_i \).

Constraints:
\[
\sum_{i=1}^{111} \text{Casting}_{C_i} \cdot x_{C_i} \leq 7650
\]
\[
\sum_{i=1}^{111} \text{Milling}_{C_i} \cdot x_{C_i} \leq 6320
\]
\[
\sum_{i=1}^{111} \text{Finishing}_{C_i} \cdot x_{C_i} \leq 5538
\]
\[
\sum_{i=1}^{111} \text{Assembly}_{C_i} \cdot x_{C_i} \leq 5957
\]
\[
\sum_{i=1}^{111} \text{QA\&Packaging}_{C_i} \cdot x_{C_i} \leq 6988
\]
where, for each workshop, the coefficient for \( x_{C_i} \) is the value in the corresponding row and column of processing_time_unit.csv.

Variable domains:
\[
x_{C_i} \in \mathbb{Z}_+, \quad \forall i = 1, ..., 111
\]

All coefficients and bounds are taken directly from the supplied CSVs, preserving their order and identifiers.

Summary:
- Decision variables: \( x_{C_i} \) (integer, \( \geq 0 \)), for each component \( C_i \).
- Objective: maximize total output value (sum of unit price times quantity for all components).
- Constraints: for each workshop, total processing time used by all produced components cannot exceed the workshop's available hours.
- All data (unit prices, processing times, total hours) are used as given, with explicit component and workshop IDs.