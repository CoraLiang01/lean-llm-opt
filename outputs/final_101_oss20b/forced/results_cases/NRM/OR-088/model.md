**Mathematical model**

```markdown
**Sets**  
\(I=\{1,\dots ,15\}\) plants (F1–F15)  
\(J=\{1,\dots ,15\}\) customers (C1–C15)

**Parameters**  
\(\text{fixed\_cost}_i\) opening cost of plant \(i\)  
\(\text{capacity}_i\) maximum units plant \(i\) can supply  
\(c_{ij}\) per‑unit transport cost from plant \(i\) to customer \(j\)  
\(d_j\) demand of customer \(j\)

**Decision variables**  
\(y_i\in\{0,1\}\) 1 if plant \(i\) is opened, 0 otherwise  
\(x_{ij}\ge 0\) units shipped from plant \(i\) to customer \(j\)

**Objective**  
\[
\min \;\sum_{i\in I}\text{fixed\_cost}_i\,y_i
      +\sum_{i\in I}\sum_{j\in J}c_{ij}\,x_{ij}
\]

**Constraints**

1. **Capacity of opened plants**  
   \[
   \sum_{j\in J}x_{ij}\;\le\;\text{capacity}_i\,y_i
   \qquad \forall i\in I
   \]

2. **Demand satisfaction**  
   \[
   \sum_{i\in I}x_{ij}\;=\;d_j
   \qquad \forall j\in J
   \]

3. **Non‑negativity**  
   \[
   x_{ij}\;\ge\;0 \qquad \forall i\in I,\;j\in J
   \]

4. **Binary opening decisions**  
   \[
   y_i\in\{0,1\}\qquad \forall i\in I
   \]
```

---

### Data Mapping

| Source | Table ID | Column | Parameter / Index |
|--------|----------|--------|-------------------|
| `cost.csv` | `file_0_view_0` | `plant` | plant index \(i\) (F1→1, …, F15→15) |
| | | `fixed_cost` | \(\text{fixed\_cost}_i\) |
| | | `capacity` | \(\text{capacity}_i\) |
| | | `C1` … `C15` | \(c_{i1}\) … \(c_{i15}\) |
| `demand.csv` | `file_1_view_0` | `customer` | customer index \(j\) (C1→1, …, C15→15) |
| | | `demand` | \(d_j\) |

*All parameter values are taken directly from the corresponding columns of the two CSV files; no additional data are introduced.*