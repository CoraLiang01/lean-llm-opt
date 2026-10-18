#### Index Sets
- $L$: set of all generation lot options for coal, gas, and renewables, as returned from energy.csv.

#### Parameters
- $c_\ell$: cost per lot for lot option $\ell \in L$ (from column "cost_per_lot", table_id: file_0_view_0).
- $g_\ell$: generation per lot for lot option $\ell \in L$ (from column "gen_per_lot", table_id: file_0_view_0).
- $D$: total demand to be met (given as 200 in the query).

#### Decision Variables
- $x_\ell \in \mathbb{Z}_+, \quad \forall \ell \in L$: number of lots of option $\ell$ to purchase (integer, non-negative).

#### Objective
\[
\min \sum_{\ell \in L} c_\ell \cdot x_\ell
\]

#### Constraints

1. **Demand Satisfaction**
   \[
   \sum_{\ell \in L} g_\ell \cdot x_\ell \geq D
   \]

2. **Lot Integrality**
   \[
   x_\ell \in \mathbb{Z}_+, \quad \forall \ell \in L
   \]

---

#### Data Mapping

- Table: energy.csv (table_id: file_0_view_0)
    - Index set $L$: all records where "tech" is exactly one of {"coal", "gas", "renewables"} (CSVQA filter applied).
    - Parameter $c_\ell$: column "cost_per_lot".
    - Parameter $g_\ell$: column "gen_per_lot".
    - Lot option identifier: column "option".
- Demand $D$ is given as 200 in the user query.

No additional constraints or data are imposed by the query. All records returned by the CSVQA filter are included in $L$.