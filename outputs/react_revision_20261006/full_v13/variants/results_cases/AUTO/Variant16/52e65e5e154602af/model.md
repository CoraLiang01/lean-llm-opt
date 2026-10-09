## Symbolic Mathematical Model

**Sets**
- $G$: set of component families (from option_catalog.csv, column "Family")
- $O_g$: set of options for family $g \in G$ (from option_catalog.csv, column "Option" for each $g$)

**Parameters** (from option_catalog.csv and resource_limits.csv)
- $v_{g,o}$: value of option $o$ in family $g$ ("Value", file_0_view_0)
- $w_{g,o}$: weight of option $o$ in family $g$ ("Weight", file_0_view_0)
- $b_{g,o}$: budget use of option $o$ in family $g$ ("BudgetUse", file_0_view_0)
- $W^{\max}$: total weight limit ("Limit" where "Resource" = "Weight", file_1_view_0)
- $B^{\max}$: total budget use limit ("Limit" where "Resource" = "BudgetUse", file_1_view_0)

**Decision Variables**
- $x_{g,o} \in \{0,1\}$: 1 if option $o$ is selected from family $g$, 0 otherwise

**Objective**
\[
\max \sum_{g \in G} \sum_{o \in O_g} v_{g,o} x_{g,o}
\]

**Constraints**
1. **Exactly one option per family:**
   \[
   \sum_{o \in O_g} x_{g,o} = 1 \qquad \forall g \in G
   \]
2. **Total weight limit:**
   \[
   \sum_{g \in G} \sum_{o \in O_g} w_{g,o} x_{g,o} \leq W^{\max}
   \]
3. **Total budget use limit:**
   \[
   \sum_{g \in G} \sum_{o \in O_g} b_{g,o} x_{g,o} \leq B^{\max}
   \]
4. **Binary variables:**
   \[
   x_{g,o} \in \{0,1\} \qquad \forall g \in G,\, o \in O_g
   \]

---

## Data Mapping

- $G$ and $O_g$: All unique "Family" and corresponding "Option" values from option_catalog.csv (table_id: file_0_view_0, columns: "Family", "Option")
- $v_{g,o}$: "Value" column in option_catalog.csv (file_0_view_0)
- $w_{g,o}$: "Weight" column in option_catalog.csv (file_0_view_0)
- $b_{g,o}$: "BudgetUse" column in option_catalog.csv (file_0_view_0)
- $W^{\max}$: "Limit" where "Resource" = "Weight" in resource_limits.csv (file_1_view_0)
- $B^{\max}$: "Limit" where "Resource" = "BudgetUse" in resource_limits.csv (file_1_view_0)
- $x_{g,o}$: binary variable for each family-option pair

All indices, parameters, and constraints are mapped directly to the current CSV data as described.