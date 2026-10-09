## Binary Multi-Choice Knapsack Model

**Sets**
- $\mathcal{F}$: set of product families (from option_catalog.csv, column Family)
- $\mathcal{O}_f$: set of options for family $f$ (from option_catalog.csv, column Option for each Family)

**Parameters** (from option_catalog.csv, table_id: file_0_view_0)
- $v_{fo}$: value of option $o$ in family $f$ (Value)
- $w_{fo}$: shelf weight of option $o$ in family $f$ (Weight)
- $b_{fo}$: budget usage of option $o$ in family $f$ (BudgetUse)

**Resource Limits** (from resource_limits.csv, table_id: file_1_view_0)
- $W^{\max}$: total shelf weight limit (Limit where Resource = Weight)
- $B^{\max}$: total budget limit (Limit where Resource = BudgetUse)

**Decision Variables**
- $x_{fo} \in \{0,1\}$: 1 if option $o$ is selected for family $f$, 0 otherwise

**Objective**
\[
\max \sum_{f \in \mathcal{F}} \sum_{o \in \mathcal{O}_f} v_{fo} x_{fo}
\]

**Constraints**
1. **One option per family:**
   \[
   \sum_{o \in \mathcal{O}_f} x_{fo} = 1 \qquad \forall f \in \mathcal{F}
   \]
2. **Shelf weight limit:**
   \[
   \sum_{f \in \mathcal{F}} \sum_{o \in \mathcal{O}_f} w_{fo} x_{fo} \leq W^{\max}
   \]
3. **Budget limit:**
   \[
   \sum_{f \in \mathcal{F}} \sum_{o \in \mathcal{O}_f} b_{fo} x_{fo} \leq B^{\max}
   \]
4. **Binary selection:**
   \[
   x_{fo} \in \{0,1\} \qquad \forall f \in \mathcal{F},\ o \in \mathcal{O}_f
   \]

---

### Data Mapping

- $\mathcal{F}$: All unique values in column Family of file_0_view_0 (option_catalog.csv)
- $\mathcal{O}_f$: All Option values for each Family $f$ in file_0_view_0
- $v_{fo}$: Value column in file_0_view_0 for each (Family, Option)
- $w_{fo}$: Weight column in file_0_view_0 for each (Family, Option)
- $b_{fo}$: BudgetUse column in file_0_view_0 for each (Family, Option)
- $W^{\max}$: Limit where Resource = Weight in file_1_view_0 (resource_limits.csv)
- $B^{\max}$: Limit where Resource = BudgetUse in file_1_view_0 (resource_limits.csv)
- $x_{fo}$: Binary variable for each (Family, Option) pair

All indices, parameters, and limits are taken directly from the current CSV data as described above.