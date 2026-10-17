## Mathematical Model

**Sets**  
\(I=\{\,i\mid i=\text{StorageID in file\_0\_view\_0}\,\}\)  
\(J=\{\,j\mid j=\text{ProductName in file\_1\_view\_0}\,\}\)

**Parameters**  
\(\text{Cap}[i]\) – capacity of storage \(i\) (from column *Capacity* of file\_0\_view\_0)  
\(\text{Val}[j]\) – value of product \(j\) (from column *Value* of file\_1\_view\_0)  
\(\text{Wgt}[j]\) – weight of product \(j\) (from column *Weight* of file\_1\_view\_0)

**Decision variables**  
\(x_{ij}\in \mathbb{Z}_{+}\) – number of units of product \(j\) placed in storage \(i\)

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Val}[j]\;x_{ij}
\]

**Constraints**  

Capacity limits for each storage:
\[
\sum_{j\in J}\text{Wgt}[j]\;x_{ij}\;\le\;\text{Cap}[i]\qquad \forall i\in I
\]

Non‑negativity and integrality:
\[
x_{ij}\;\ge\;0,\; x_{ij}\in\mathbb{Z}\qquad \forall i\in I,\; j\in J
\]

---

## Data Mapping Rules

| Parameter | Source Table | Source Column | Index Column | Notes |
|-----------|--------------|---------------|--------------|-------|
| \(\text{Cap}[i]\) | file\_0\_view\_0 | Capacity | StorageID | One value per storage area |
| \(\text{Val}[j]\) | file\_1\_view\_0 | Value | ProductName | One value per product type |
| \(\text{Wgt}[j]\) | file\_1\_view\_0 | Weight | ProductName | One weight per product type |

All indices \(i\) and \(j\) are taken from the complete set of rows returned in the respective tables; no additional filtering is applied. The model uses only the parameters and indices explicitly defined above.