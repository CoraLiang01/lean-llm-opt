## Binary Multi-Choice Knapsack Model

**Sets:**
- $F$: set of product families (from column "Family" in option_catalog.csv)
- $O_f$: set of options for family $f$ (from column "Option" in option_catalog.csv, grouped by "Family")

**Parameters (from option_catalog.csv and resource_limits.csv):**
- $v_{fo}$: value of option $o$ in family $f$ ("Value", table_id: file_0_view_0)
- $w_{fo}$: shelf weight of option $o$ in family $f$ ("Weight", table_id: file_0_view_0)
- $b_{fo}$: budget usage of option $o$ in family $f$ ("BudgetUse", table_id: file_0_view_0)
- $W^{\max}$: total shelf weight limit ("Limit" where "Resource" = "Weight", table_id: file_1_view_0)
- $B^{\max}$: total budget limit ("Limit" where "Resource" = "BudgetUse", table_id: file_1_view_0)

**Decision Variables:**
- $x_{fo} \in \{0,1\}$: 1 if option $o$ is selected for family $f$, 0 otherwise

**Objective:**
\[
\max \sum_{f \in F} \sum_{o \in O_f} v_{fo} x_{fo}
\]

**Constraints:**
1. **One option per family:**
   \[
   \sum_{o \in O_f} x_{fo} = 1 \quad \forall f \in F
   \]
2. **Shelf weight limit:**
   \[
   \sum_{f \in F} \sum_{o \in O_f} w_{fo} x_{fo} \leq W^{\max}
   \]
3. **Budget limit:**
   \[
   \sum_{f \in F} \sum_{o \in O_f} b_{fo} x_{fo} \leq B^{\max}
   \]
4. **Binary selection:**
   \[
   x_{fo} \in \{0,1\} \quad \forall f \in F,\, o \in O_f
   \]

---

### Data Mapping

- $F$, $O_f$, $v_{fo}$, $w_{fo}$, $b_{fo}$: from option_catalog.csv (table_id: file_0_view_0, columns "Family", "Option", "Value", "Weight", "BudgetUse")
- $W^{\max}$: from resource_limits.csv (table_id: file_1_view_0, column "Limit", row where "Resource" = "Weight")
- $B^{\max}$: from resource_limits.csv (table_id: file_1_view_0, column "Limit", row where "Resource" = "BudgetUse")