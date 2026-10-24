## Mathematical Model

**Sets**  
\(I=\{1,\dots ,15\}\) (storage areas)  
\(J=\{1,\dots ,10\}\) (product types)

**Parameters**  
\(\text{Cap}[i]\) capacity of storage area \(i\) (from *capacity.csv*, column *Capacity*)  
\(\text{Val}[j]\) value of product \(j\) (from *products.csv*, column *Value*)  
\(\text{Wgt}[j]\) size (weight) of product \(j\) (from *products.csv*, column *Weight*)

**Decision variables**  
\(x_{ij}\in \mathbb{Z}_{+}\) number of units of product \(j\) placed in storage area \(i\)

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Val}[j]\;x_{ij}
\]

**Constraints**  

Capacity limits for each storage area  
\[
\sum_{j\in J}\text{Wgt}[j]\;x_{ij}\;\le\;\text{Cap}[i]\qquad \forall i\in I
\]

Non‑negativity and integrality  
\[
x_{ij}\;\ge\;0,\; x_{ij}\in \mathbb{Z}\qquad \forall i\in I,\; j\in J
\]

---

## Data Mapping

| Parameter | Source Table | Column | Mapping Rule |
|-----------|--------------|--------|--------------|
| \(\text{Cap}[i]\) | `file_0_view_0` | `Capacity` | For each record, map `StorageID` → index \(i\); assign `Capacity` to \(\text{Cap}[i]\). |
| \(\text{Val}[j]\) | `file_1_view_0` | `Value` | For each record, map `ProductName` → index \(j\); assign `Value` to \(\text{Val}[j]\). |
| \(\text{Wgt}[j]\) | `file_1_view_0` | `Weight` | For each record, map `ProductName` → index \(j\); assign `Weight` to \(\text{Wgt}[j]\). |

*Indices \(i\) and \(j\) are taken directly from the `StorageID` and `ProductName` columns, respectively, without any filtering or aggregation.*