## Mathematical Model

**Sets**  
\(I=\{1,\dots ,15\}\) (storage areas)  
\(J=\{1,\dots ,10\}\) (product types)

**Parameters**  
\(\text{Capacity}[i]\) – capacity of storage area \(i\) (units of weight)  
\(\text{Value}[j]\) – monetary value of one unit of product \(j\)  
\(\text{Weight}[j]\) – weight of one unit of product \(j\)

**Decision variables**  
\(x_{ij}\in \mathbb{Z}_{+}\) – number of units of product \(j\) placed in storage area \(i\)

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Value}[j]\;x_{ij}
\]

**Constraints**  
Capacity limits for each storage area  
\[
\sum_{j\in J}\text{Weight}[j]\;x_{ij}\;\le\;\text{Capacity}[i]\qquad \forall i\in I
\]

All variables are non‑negative integers  
\[
x_{ij}\;\ge\;0,\; x_{ij}\in \mathbb{Z}\qquad \forall i\in I,\; j\in J
\]

---

## Data Mapping

| Parameter | Source Table | Column | Mapping Rule |
|-----------|--------------|--------|--------------|
| \(\text{Capacity}[i]\) | `file_0_view_0` | `Capacity` | For each record, map `StorageID` → index \(i\); assign `Capacity` value. |
| \(\text{Value}[j]\) | `file_1_view_0` | `Value` | For each record, map `ProductName` → index \(j\); assign `Value` value. |
| \(\text{Weight}[j]\) | `file_1_view_0` | `Weight` | For each record, map `ProductName` → index \(j\); assign `Weight` value. |

Indices \(i\) and \(j\) are taken from the full set of records in the respective tables; no additional filtering is applied. The model uses the exact numeric values as provided in the source tables.