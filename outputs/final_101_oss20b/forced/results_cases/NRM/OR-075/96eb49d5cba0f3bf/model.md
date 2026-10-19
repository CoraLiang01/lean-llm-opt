**Abstract Model**

**Index Set**  
\( I = \{1,2,\dots ,110\} \)

**Parameters**  
\( \text{cost}_i \) – capital investment required for project \( i \) (k$)  
\( \text{npv}_i \) – expected Net Present Value of project \( i \) (k$)  

**Decision Variables**  
\( x_i \in \{0,1\} \) – 1 if project \( i \) is selected, 0 otherwise, \( \forall i \in I \)

**Objective Function**  
\( \displaystyle \max \sum_{i \in I} \text{npv}_i \, x_i \)

**Constraints**

1. **Budget Constraint**  
\( \displaystyle \sum_{i \in I} \text{cost}_i \, x_i \;\le\; 1000 \)

2. **Mutually Exclusive Constraint (Projects 4 & 7)**  
\( \displaystyle x_4 + x_7 \;\le\; 1 \)

3. **Pre‑requisite Constraint (Project 6 requires 1)**  
\( \displaystyle x_6 \;\le\; x_1 \)

4. **Contingent Constraint (Project 10 requires 5)**  
\( \displaystyle x_{10} \;\le\; x_5 \)

5. **Variable Domain**  
\( \displaystyle x_i \in \{0,1\}, \quad \forall i \in I \)

---

**Data Mapping**

| Source | Table ID | Column Names |
|--------|----------|--------------|
| Project Data | `file_0_view_0` | `Project ID`, `Capital (k$)`, `NPV (k$)` |

*The parameters `cost_i` and `npv_i` are mapped directly from the `Capital (k$)` and `NPV (k$)` columns of `file_0_view_0` using the `Project ID` as the index.*