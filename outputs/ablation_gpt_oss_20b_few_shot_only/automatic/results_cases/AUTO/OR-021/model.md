**Symbolic Model**

```markdown
**Sets**  
\(I\) – set of all products (rows of the CSV file)

**Parameters**  
\(\text{Revenue}_i\) – revenue per unit of product \(i\)  
\(\text{Demand}_i\) – deterministic demand for product \(i\)  
\(\text{Inventory}_i\) – initial inventory of product \(i\)

**Decision Variables**  
\(x_i \in \mathbb{Z}_{+}\) – number of units of product \(i\) to fulfill  
(\(x_i \ge 1\))

**Objective**  
\[
\max \;\sum_{i\in I} \text{Revenue}_i \; x_i
\]

**Constraints**  

1. **Demand limit**  
\[
x_i \;\le\; \text{Demand}_i \qquad \forall i\in I
\]

2. **Inventory limit**  
\[
x_i \;\le\; \text{Inventory}_i \qquad \forall i\in I
\]

3. **Positive integer fulfillment**  
\[
x_i \;\ge\; 1 \qquad \forall i\in I
\]
```

---

**Data Mapping**

| Parameter | Source File | Source Column |
|-----------|-------------|---------------|
| \(\text{Revenue}_i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM12/Salesofsummerclothes.csv` | `Revenue` |
| \(\text{Demand}_i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM12/Salesofsummerclothes.csv` | `Demand` |
| \(\text{Inventory}_i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM12/Salesofsummerclothes.csv` | `Initial Inventory` |
| Index set \(I\) | Same file | Each row (product) |
