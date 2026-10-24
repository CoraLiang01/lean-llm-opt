**Mathematical Model**

Let  

- \(M=\{M_1,\dots ,M_{12}\}\) (“Machine” rows)  
- \(T=\{A,B,C,D,E,F,G,H,I,J,K,L\}\) (“Task” columns)  

Parameters  

- \(c_{i,j}\) machining cost for assigning machine \(i\in M\) to task \(j\in T\).

Decision variables  

- \(x_{i,j}\in\{0,1\}\) \(x_{i,j}=1\) if machine \(i\) is assigned to task \(j\), otherwise \(0\).

Objective  

\[
\min \sum_{i\in M}\sum_{j\in T} c_{i,j}\,x_{i,j}
\]

Constraints  

1. **Machine assignment** \(\displaystyle \sum_{j\in T} x_{i,j}=1 \quad \forall i\in M\)

2. **Task assignment** \(\displaystyle \sum_{i\in M} x_{i,j}=1 \quad \forall j\in T\)

3. **Binary variables** \(x_{i,j}\in\{0,1\}\quad \forall i\in M,\; j\in T\)

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_{i,j}\) | `cost_12x12.csv` (table_id `file_0_view_0`) | For each record: <br>• `Machine` column value → index \(i\) <br>• Each column header \(j\in T\) → index \(j\) <br>• Cell value → \(c_{i,j}\) |

No numerical values are enumerated; the mapping references the CSV structure directly.