## Mathematical Model

**Sets**  
\(I=\{S1,S2,S3\}\) (warehouses)  
\(J=\{C1,C2,C3\}\) (customers)

**Parameters**  
\(f_i\) fixed cost of warehouse \(i\) (from *fixed_cost.csv*, column *fixed_costs*)  
\(t_{ij}\) transportation cost per unit from warehouse \(i\) to customer \(j\) (from *transportation_costs.csv*)  
\(d_j\) demand of customer \(j\) (from *demand.csv*, column *demand*)

**Decision Variables**  
\(y_i\in\{0,1\}\) warehouse \(i\) is activated  
\(x_{ij}\ge 0\) quantity supplied from warehouse \(i\) to customer \(j\)

**Objective**  
\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij}
\]

**Constraints**  

1. **Demand satisfaction**  
\[
\sum_{i\in I} x_{ij} \;\ge\; d_j \qquad \forall j\in J
\]

2. **Non‑negativity**  
\[
x_{ij}\;\ge\;0 \qquad \forall i\in I,\; j\in J
\]

3. **Binary activation**  
\[
y_i\in\{0,1\} \qquad \forall i\in I
\]

*(No capacity or linking constraints are specified in the problem statement.)*

---

## Data Mapping

| Parameter | Source Table | Source Column | Notes |
|-----------|--------------|---------------|-------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | Row identifier `Unnamed: 0` gives warehouse id |
| \(t_{ij}\) | `file_2_view_0` | `C1`, `C2`, `C3` | Row identifier `Unnamed: 0` gives warehouse id; column names give customer id |
| \(d_j\) | `file_0_view_0` | `demand` | Row identifier `customer` gives customer id |

All indices \(i\) and \(j\) are taken from the full set of rows in the corresponding tables, as no further filtering is requested.