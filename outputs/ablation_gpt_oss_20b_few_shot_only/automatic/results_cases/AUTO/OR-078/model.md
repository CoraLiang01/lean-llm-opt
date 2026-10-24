**Mathematical model**

Let  

* \(I\) – set of all lot options listed in *energy.csv*  
* \(x_i\) – number of lots of option \(i\) ordered (integer, \(x_i\ge 0\))

Parameters (extracted from *energy.csv*)  

| Symbol | Source column | File path |
|--------|---------------|-----------|
| \(\text{gen}_{i}\) | `gen_per_lot` | `Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv` |
| \(\text{cost}_{i}\) | `cost_per_lot` | same file |

Demand requirement  

* \(D = 200\) (kg of generation)

The optimization problem is

\[
\begin{aligned}
\min_{x}\quad & \sum_{i\in I} \text{cost}_{i}\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} \text{gen}_{i}\,x_i \;\ge\; D \\[4pt]
& x_i \in \mathbb{Z}_{+}\quad \forall i\in I
\end{aligned}
\]

---

**Data‑mapping rules**

1. **Index set**  
   *\(I\)* is the set of all rows in *energy.csv*.  
   Each row is identified by its `option` value (e.g., `coal_001`, `gas_001`, `renewables_001`, …).

2. **Parameters**  
   * For every \(i\in I\):  
     * \(\text{gen}_{i}\) ← value in column `gen_per_lot` of the row with `option = i`.  
     * \(\text{cost}_{i}\) ← value in column `cost_per_lot` of the same row.

3. **Demand**  
   * \(D = 200\) (constant, not read from a file).

4. **Decision variables**  
   * \(x_i\) is defined for each \(i\in I\).  
   * Domain: integer, non‑negative.

5. **Constraints**  
   * The single resource constraint uses the mapped \(\text{gen}_{i}\) values.  
   * No other constraints are imposed.

The model above is a pure integer linear program that satisfies all requirements stated in the query and uses only the data provided in *energy.csv*.