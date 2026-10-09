## Symbolic Mathematical Model

**Sets:**
- $G$: set of families (from option_catalog.csv, column "Family")
- $O_g$: set of options for family $g \in G$ (from option_catalog.csv, column "Option" for each $g$)
- $R$: set of resources (from resource_limits.csv, column "Resource")

**Parameters:**
- $v_{g,o}$: value of option $o$ in family $g$ (option_catalog.csv, "Value", table_id=file_0_view_0)
- $w_{g,o}$: weight of option $o$ in family $g$ (option_catalog.csv, "Weight", table_id=file_0_view_0)
- $b_{g,o}$: budget use of option $o$ in family $g$ (option_catalog.csv, "BudgetUse", table_id=file_0_view_0)
- $W^{\max}$: total weight limit (resource_limits.csv, row with Resource="Weight", "Limit", table_id=file_1_view_0)
- $B^{\max}$: total budget use limit (resource_limits.csv, row with Resource="BudgetUse", "Limit", table_id=file_1_view_0)

**Decision Variables:**
- $x_{g,o} \in \{0,1\}$: $=1$ if option $o$ is selected from family $g$, $0$ otherwise

**Objective:**
\[
\max \sum_{g \in G} \sum_{o \in O_g} v_{g,o} \, x_{g,o}
\]

**Constraints:**

1. **Exactly one option per family:**
   \[
   \sum_{o \in O_g} x_{g,o} = 1 \qquad \forall g \in G
   \]

2. **Total weight constraint:**
   \[
   \sum_{g \in G} \sum_{o \in O_g} w_{g,o} \, x_{g,o} \leq W^{\max}
   \]

3. **Total budget use constraint:**
   \[
   \sum_{g \in G} \sum_{o \in O_g} b_{g,o} \, x_{g,o} \leq B^{\max}
   \]

4. **Binary variables:**
   \[
   x_{g,o} \in \{0,1\} \qquad \forall g \in G,\, o \in O_g
   \]

---

### Data Mapping

- $G$, $O_g$, $v_{g,o}$, $w_{g,o}$, $b_{g,o}$: from option_catalog.csv (table_id=file_0_view_0, columns "Family", "Option", "Value", "Weight", "BudgetUse")
- $W^{\max}$: from resource_limits.csv (table_id=file_1_view_0, row Resource="Weight", column "Limit")
- $B^{\max}$: from resource_limits.csv (table_id=file_1_view_0, row Resource="BudgetUse", column "Limit")