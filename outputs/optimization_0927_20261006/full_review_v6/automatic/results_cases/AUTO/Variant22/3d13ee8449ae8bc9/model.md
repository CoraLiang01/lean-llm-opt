Let $x_{co}$ be a binary variable equal to 1 if option $o$ from family $c$ is selected, 0 otherwise.

**Parameters:**

- For each family-option pair $(c,o)$, let $v_{co}$ be the Value, $w_{co}$ the Weight, and $l_{co}$ the LaborHours from option_catalog.csv.
- Let $W^{\max} = 55$ and $L^{\max} = 64$ be the total limits for Weight and LaborHours, respectively.

**Model:**

Maximize total value:
$$
\max \sum_{(c,o)} v_{co} \, x_{co}
$$

Subject to:

**1. Exactly one option per family:**

For each family $c$:
$$
\sum_{o} x_{co} = 1
$$

Specifically:
- $\sum_{o} x_{C1,o} = 1$
- $\sum_{o} x_{C2,o} = 1$
- $\sum_{o} x_{C3,o} = 1$
- $\sum_{o} x_{C4,o} = 1$
- $\sum_{o} x_{C5,o} = 1$
- $\sum_{o} x_{C6,o} = 1$

**2. Total weight constraint:**
$$
\sum_{(c,o)} w_{co} \, x_{co} \leq 55
$$

**3. Total labor-hour constraint:**
$$
\sum_{(c,o)} l_{co} \, x_{co} \leq 64
$$

**4. Binary restrictions:**
$$
x_{co} \in \{0,1\} \qquad \forall\, (c,o)
$$

**Data (from option_catalog.csv and resource_limits.csv, in source order):**

| Family | Option | Value | Weight | LaborHours |
|--------|--------|-------|--------|------------|
| C1     | O1     | 18    | 5      | 6          |
| C1     | O2     | 27    | 8      | 9          |
| C1     | O3     | 30    | 10     | 11         |
| C2     | O1     | 24    | 7      | 8          |
| C2     | O2     | 32    | 11     | 13         |
| C2     | O3     | 28    | 9      | 10         |
| C3     | O1     | 22    | 6      | 7          |
| C3     | O2     | 35    | 12     | 14         |
| C3     | O3     | 31    | 10     | 12         |
| C4     | O1     | 20    | 5      | 8          |
| C4     | O2     | 29    | 9      | 11         |
| C4     | O3     | 34    | 12     | 13         |
| C5     | O1     | 23    | 7      | 7          |
| C5     | O2     | 33    | 11     | 12         |
| C5     | O3     | 26    | 8      | 10         |
| C6     | O1     | 21    | 6      | 6          |
| C6     | O2     | 30    | 10     | 11         |
| C6     | O3     | 36    | 13     | 15         |

Resource limits:
- Weight $\leq 55$
- LaborHours $\leq 64$

**Variables:**
- $x_{co} \in \{0,1\}$ for each family-option pair as listed above.