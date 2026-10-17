**Mathematical Model**

```markdown
**Sets**  
$P$ – set of widgets (Widget1,…,Widget141)

**Parameters**  
$\text{LaborHours}_p,\;\text{MaterialA}_p,\;\text{MaterialB}_p,\;\text{Profit}_p \;\; (p\in P)$  
$L_{\text{LaborHours}},\;L_{\text{MaterialA}},\;L_{\text{MaterialB}}$ – monthly limits  
$\text{CatalystRate}_p$ – kg of CatalystX per unit of widget $p$  
$\text{SaleCap}=1500$ – kg (maximum monthly sale)  
$\text{SalePrice}=300$ – $/kg$  
$\text{DisposalCost}=200$ – $/kg$

**Decision Variables**  
$x_p \ge 0 \;\; (p\in P)$ – units of widget $p$ produced  
$s \ge 0$ – kg of CatalystX sold

**Objective**  
\[
\max \;\sum_{p\in P}\text{Profit}_p\,x_p
      \;-\;\text{DisposalCost}\sum_{p\in P}\text{CatalystRate}_p\,x_p
      \;+\;(\text{SalePrice}+\text{DisposalCost})\,s
\]
(Only Widget3 has $\text{CatalystRate}_{\text{Widget3}}=5$, all others $0$.)

**Constraints**  

1. **Labor**  
\[
\sum_{p\in P}\text{LaborHours}_p\,x_p \;\le\; L_{\text{LaborHours}}
\]

2. **Material A**  
\[
\sum_{p\in P}\text{MaterialA}_p\,x_p \;\le\; L_{\text{MaterialA}}
\]

3. **Material B**  
\[
\sum_{p\in P}\text{MaterialB}_p\,x_p \;\le\; L_{\text{MaterialB}}
\]

4. **Catalyst sale – cannot exceed production**  
\[
s \;\le\; \sum_{p\in P}\text{CatalystRate}_p\,x_p
\]

5. **Catalyst sale – market cap**  
\[
s \;\le\; \text{SaleCap}
\]

6. **Non‑negativity**  
\[
x_p \ge 0 \;\;(p\in P), \qquad s \ge 0
\]
```

---

**Data Mapping Rules**

| Table ID | Source Columns | Mapping to Model |
|----------|----------------|------------------|
| `file_0_view_0` | `Resource`, `MonthlyLimit` | For each row: `L_[Resource] = MonthlyLimit` (e.g., `L_LaborHours = 5000`) |
| `file_1_view_0` | `Product`, `LaborHours`, `MaterialA`, `MaterialB`, `Profit` | For each row: <br>• `P` includes `Product` <br>• `LaborHours_[Product] = LaborHours` <br>• `MaterialA_[Product] = MaterialA` <br>• `MaterialB_[Product] = MaterialB` <br>• `Profit_[Product] = Profit` |
| **Derived Parameter** | – | `CatalystRate_[Product] = 5` if `Product = 'Widget3'`, else `0` |
| **Constants** | – | `SaleCap = 1500`, `SalePrice = 300`, `DisposalCost = 200` |

All numeric values are taken as floats from the CSV files; identifiers (`Product`, `Resource`) are treated as categorical indices. No additional data or aggregation is introduced beyond what is supplied.