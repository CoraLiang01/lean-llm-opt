**Mathematical Model**

Let  

* \(I=\{S1,S2\}\) – set of suppliers  
* \(J=\{C1,C2\}\) – set of supermarkets  

Parameters (all taken directly from the supplied tables)

| Parameter | Symbol | Source table | Column(s) |
|-----------|--------|--------------|-----------|
| Fixed cost of supplier \(i\) | \(f_i\) | `fixed_cost.csv` | `fixed_costs` |
| Transportation cost per unit from supplier \(i\) to supermarket \(j\) | \(c_{ij}\) | `transportation_costs.csv` | `C1`, `C2` |
| Demand of supermarket \(j\) | \(d_j\) | `demand.csv` | `demand` |

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if supplier \(i\) is activated, 0 otherwise.  
* \(x_{ij}\ge 0\) – quantity supplied from supplier \(i\) to supermarket \(j\).

---

### Objective
\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

### Constraints

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;=\; d_j \qquad \forall j\in J
   \]

2. **Supplier activation linkage**  
   \[
   x_{ij} \;\le\; d_j\,y_i \qquad \forall i\in I,\; j\in J
   \]
   (If \(y_i=0\) then \(x_{ij}=0\); if \(y_i=1\) the quantity can be up to the demand of \(j\).)

3. **Non‑negativity**  
   \[
   x_{ij}\;\ge\;0 \qquad \forall i\in I,\; j\in J
   \]

4. **Binary activation**  
   \[
   y_i \in \{0,1\} \qquad \forall i\in I
   \]

---

**Data Mapping**

| Table ID | Column | Parameter | Notes |
|----------|--------|-----------|-------|
| `file_0_view_0` | `customer` | \(j\) | Supermarket identifiers |
| `file_0_view_0` | `demand` | \(d_j\) | Demand per supermarket |
| `file_1_view_0` | `Unnamed: 0` | \(i\) | Supplier identifiers |
| `file_1_view_0` | `fixed_costs` | \(f_i\) | Fixed activation cost |
| `file_2_view_0` | `Unnamed: 0` | \(i\) | Supplier identifiers (same as above) |
| `file_2_view_0` | `C1`, `C2` | \(c_{ij}\) | Transportation cost per unit from supplier \(i\) to supermarket \(j\) |

All parameters are taken verbatim from the corresponding columns; no additional scaling or transformation is applied. The model uses the full set of suppliers and supermarkets returned by the data extraction.