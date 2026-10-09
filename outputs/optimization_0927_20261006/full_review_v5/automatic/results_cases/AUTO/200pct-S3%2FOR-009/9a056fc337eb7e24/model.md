**Sets and Indices:**

Let $I$ be the set of areas (indexed by $i$), with the following members in source order:
- Queens
- Brooklyn
- Manhattan
- Bronx
- Staten Island
- Harlem
- Upper East Side
- Lower Manhattan
- Midtown
- Long Island City
- Williamsburg
- Bushwick
- Flatbush
- Greenpoint
- Park Slope
- Astoria
- Jackson Heights
- Flushing
- Sunnyside
- Ditmars

**Parameters:**

For each area $i$:
- $v_i$ = Value (development benefit per unit)
- $w_i$ = Weight (resource requirement per unit)

From capacity.csv:
- $C$ = 586 (overall development capacity)

| Area               | $v_i$ (Value) | $w_i$ (Weight) |
|--------------------|:-------------:|:--------------:|
| Queens             | 469           | 954            |
| Brooklyn           | 290           | 650            |
| Manhattan          | 236           | 961            |
| Bronx              | 235           | 950            |
| Staten Island      | 745           | 379            |
| Harlem             | 684           | 776            |
| Upper East Side    | 444           | 381            |
| Lower Manhattan    | 172           | 808            |
| Midtown            | 1000          | 937            |
| Long Island City   | 336           | 608            |
| Williamsburg       | 546           | 912            |
| Bushwick           | 535           | 391            |
| Flatbush           | 539           | 465            |
| Greenpoint         | 831           | 490            |
| Park Slope         | 139           | 918            |
| Astoria            | 432           | 787            |
| Jackson Heights    | 627           | 347            |
| Flushing           | 629           | 274            |
| Sunnyside          | 292           | 642            |
| Ditmars            | 978           | 130            |

**Decision Variables:**

For each area $i$:
- $x_i \in \mathbb{Z}_{\geq 0}$: scale of development per day in area $i$

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} w_i x_i \leq 586
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

**Parameter Table (source order):**

| Area               | $v_i$ | $w_i$ |
|--------------------|-------|-------|
| Queens             | 469   | 954   |
| Brooklyn           | 290   | 650   |
| Manhattan          | 236   | 961   |
| Bronx              | 235   | 950   |
| Staten Island      | 745   | 379   |
| Harlem             | 684   | 776   |
| Upper East Side    | 444   | 381   |
| Lower Manhattan    | 172   | 808   |
| Midtown            | 1000  | 937   |
| Long Island City   | 336   | 608   |
| Williamsburg       | 546   | 912   |
| Bushwick           | 535   | 391   |
| Flatbush           | 539   | 465   |
| Greenpoint         | 831   | 490   |
| Park Slope         | 139   | 918   |
| Astoria            | 432   | 787   |
| Jackson Heights    | 627   | 347   |
| Flushing           | 629   | 274   |
| Sunnyside          | 292   | 642   |
| Ditmars            | 978   | 130   |

**Capacity:**
- $C = 586$

**Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$ for all $i$ (areas listed above)