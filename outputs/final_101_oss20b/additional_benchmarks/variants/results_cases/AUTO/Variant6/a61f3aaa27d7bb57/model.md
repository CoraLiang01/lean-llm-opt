**Mathematical Model**

Let  

* \(F\) – set of product families (obtained from *option_catalog.csv*).  
* \(O_f\) – set of options available for family \(f\in F\).  

Decision variables  

\[
x_{f,o}\in\{0,1\}\qquad \forall f\in F,\; o\in O_f
\]

Objective  

\[
\max \;\sum_{f\in F}\sum_{o\in O_f} \text{Value}_{f,o}\;x_{f,o}
\]

Constraints  

1. **One option per family**  

\[
\sum_{o\in O_f} x_{f,o}=1 \qquad \forall f\in F
\]

2. **Weight limit**  

\[
\sum_{f\in F}\sum_{o\in O_f} \text{Weight}_{f,o}\;x_{f,o}\;\le\;55
\]

3. **Budget limit**  

\[
\sum_{f\in F}\sum_{o\in O_f} \text{BudgetUse}_{f,o}\;x_{f,o}\;\le\;70
\]

All variables are binary: \(x_{f,o}\in\{0,1\}\).

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(F\) | Families | file_0_view_0 | `Family` |
| \(O_f\) | Options per family | file_0_view_0 | `Option` |
| \(\text{Value}_{f,o}\) | Option value | file_0_view_0 | `Value` |
| \(\text{Weight}_{f,o}\) | Option shelf weight | file_0_view_0 | `Weight` |
| \(\text{BudgetUse}_{f,o}\) | Option budget usage | file_0_view_0 | `BudgetUse` |
| Weight limit | Resource limit | file_1_view_0 | `Limit` where `Resource`=`Weight` | 55 |
| Budget limit | Resource limit | file_1_view_0 | `Limit` where `Resource`=`BudgetUse` | 70 |

No other data are introduced; all coefficients and limits are taken directly from the supplied tables.