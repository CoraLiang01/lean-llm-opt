**Mathematical model**

Let  

* \(I=\{S1,\dots ,S10\}\) – source locations  
* \(J=\{D1,\dots ,D20\}\) – destination locations  

Parameters  

* \(c_{ij}\) – unit transportation cost from source \(i\) to destination \(j\)  
* \(s_i\) – available supply units at source \(i\)  
* \(d_j\) – required demand units at destination \(j\)  

Decision variables  

* \(x_{ij}\ge 0\) – units shipped from \(i\) to \(j\) (continuous)  
* \(y_{ij}\in \mathbb{Z}_{+}\) – number of trucks dispatched on route \(i\to j\)  

**Objective**

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Supply limits**  
   \[
   \sum_{j\in J} x_{ij}\;\le\; s_i ,\qquad \forall i\in I
   \]

2. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}\;\ge\; d_j ,\qquad \forall j\in J
   \]

3. **Truck capacity**  
   \[
   x_{ij}\;\le\; 10\,y_{ij},\qquad \forall i\in I,\; j\in J
   \]

4. **Variable domains**  
   \[
   x_{ij}\ge 0,\; y_{ij}\in \mathbb{Z}_{+}\quad \forall i\in I,\; j\in J
   \]

---

**Data mapping rules**

| Parameter | Source table | Column mapping |
|-----------|--------------|----------------|
| \(c_{ij}\) | `file_0_view_0` (expanded_cost_matrix.csv) | Row: `source_id` → \(i\); Column: `D1`–`D20` → \(j\) |
| \(s_i\) | `file_2_view_0` (expanded_sources.csv) | `source_id` → \(i\); `supply_units` → \(s_i\) |
| \(d_j\) | `file_1_view_0` (expanded_destinations.csv) | `destination_id` → \(j\); `demand_units` → \(d_j\) |

All indices \(i\) and \(j\) are taken directly from the `source_id` and `destination_id` values in the respective tables. No additional aggregation or transformation is performed.