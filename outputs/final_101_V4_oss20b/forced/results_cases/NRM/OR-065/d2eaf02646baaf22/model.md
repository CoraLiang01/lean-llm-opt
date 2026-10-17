**Mathematical Model**

```markdown
**Sets**  
\(I\) – set of warehouses (indexed by \(i\))  
\(J\) – set of customers (indexed by \(j\))

**Parameters**  
\(\text{fixed\_cost}_i\) – fixed opening cost of warehouse \(i\)  
\(\text{demand}_j\) – total demand of customer \(j\)  
\(\text{transport\_cost}_{ij}\) – unit transportation cost from warehouse \(i\) to customer \(j\)

**Decision Variables**  
\(y_i \in \{0,1\}\) (warehouse \(i\) is opened)  
\(x_{ij} \ge 0\) (quantity supplied from \(i\) to \(j\))

**Objective**  
\[
\min Z = \sum_{i\in I} \text{fixed\_cost}_i\, y_i
          + \sum_{i\in I}\sum_{j\in J} \text{transport\_cost}_{ij}\, x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;\ge\; \text{demand}_j
   \quad \forall j\in J
   \]

2. **Linking (no supply from closed warehouses)**  
   \[
   x_{ij} \;\le\; \text{demand}_j \, y_i
   \quad \forall i\in I,\; j\in J
   \]

3. **Non‑negativity**  
   \[
   x_{ij} \;\ge\; 0
   \quad \forall i\in I,\; j\in J
   \]
```

---

### Data Mapping

| Parameter / Variable | Source Table | Source Column | Notes |
|----------------------|--------------|---------------|-------|
| \(I\) (warehouse index) | `file_1_view_0` (fixed_cost.csv) | `Unnamed: 0` | Warehouse identifiers |
| \(J\) (customer index) | `file_0_view_0` (demand.csv) | `customer` | Customer identifiers |
| \(\text{fixed\_cost}_i\) | `file_1_view_0` | `fixed_costs` | Fixed cost per warehouse |
| \(\text{demand}_j\) | `file_0_view_0` | `demand` | Demand per customer |
| \(\text{transport\_cost}_{ij}\) | `file_2_view_0` (transportation_costs.csv) | Columns `C1`, `C2`, `C3`, … (matching customers) | Unit transport cost from warehouse \(i\) to customer \(j\) |
| \(y_i\) | — | — | Binary decision variable |
| \(x_{ij}\) | — | — | Continuous (non‑negative) decision variable |

*All indices \(i\) and \(j\) are taken from the corresponding identifier columns in the source tables. No additional numeric values are embedded in the model; the parameters are linked directly to the CSV data as shown.*