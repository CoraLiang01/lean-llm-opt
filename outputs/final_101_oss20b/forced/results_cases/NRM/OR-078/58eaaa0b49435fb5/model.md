##### Abstract Model

**Index Sets**  
$L$: set of all lot identifiers in the table `energy.csv`.

**Parameters**  
$\displaystyle s_l \;:\;$ generation capacity per lot $l$ (from column `gen_per_lot`).  
$\displaystyle c_l \;:\;$ procurement cost per lot $l$ (from column `cost_per_lot`).  
$D \;=\;200$  (total demand to be satisfied).

**Decision Variables**  
$\displaystyle x_l \;:\;$ number of lots of type $l$ to purchase.  
$\displaystyle x_l \in \mathbb{Z}_{\ge 0}, \quad \forall l \in L$.

**Objective Function**  
$\displaystyle \min \;\sum_{l \in L} c_l \, x_l$

**Constraints**  

1. **Demand Satisfaction**  
$\displaystyle \sum_{l \in L} s_l \, x_l \;\ge\; D$

2. **Non‑negativity and Integrality**  
$\displaystyle x_l \in \mathbb{Z}_{\ge 0}, \quad \forall l \in L$

---

#### Data Mapping

| Source | Table ID | Column Names |
|--------|----------|--------------|
| Energy lot data | `file_0_view_0` | `option` (lot identifier), `tech` (generation type), `gen_per_lot` (s_l), `cost_per_lot` (c_l) |

*The set $L$ is derived from the distinct values in the `option` column of `file_0_view_0`.  Parameters $s_l$ and $c_l$ are taken directly from the `gen_per_lot` and `cost_per_lot` columns, respectively.*