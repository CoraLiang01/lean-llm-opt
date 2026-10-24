## Mathematical Model

**Sets**  
\(I=\{1,\dots ,14\}\) display areas (indices from *capacity.csv*)  
\(J=\{1,\dots ,20\}\) boat types (indices from *products.csv*)

**Parameters**  
\(\text{Cap}_i\) capacity of area \(i\) (from *capacity.csv*, column *Capacity*)  
\(\text{Val}_j\) value of boat type \(j\) (from *products.csv*, column *Value*)  
\(\text{Wgt}_j\) weight of boat type \(j\) (from *products.csv*, column *Weight*)

**Decision Variables**  
\(x_{ij}\in \mathbb{Z}_{\ge 0}\) number of units of boat type \(j\) placed in area \(i\)

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Val}_j\,x_{ij}
\]

**Constraints**  
Capacity of each area  
\[
\sum_{j\in J}\text{Wgt}_j\,x_{ij}\;\le\;\text{Cap}_i
\qquad \forall\, i\in I
\]

All variables are non‑negative integers (already enforced by the domain).

---

## Data Mapping

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(\text{Cap}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{Val}_j\) | `file_1_view_0` | `Value` |
| \(\text{Wgt}_j\) | `file_1_view_0` | `Weight` |

| Variable | Domain |
|----------|--------|
| \(x_{ij}\) | integer, \(x_{ij}\ge 0\) |

The index sets \(I\) and \(J\) are defined by all rows returned in the respective tables (`file_0_view_0` and `file_1_view_0`). No additional filtering or aggregation is applied.