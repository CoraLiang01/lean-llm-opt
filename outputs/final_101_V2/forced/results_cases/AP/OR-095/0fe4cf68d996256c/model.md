##### Decision Variables

Let $x_i$ denote the number of units produced of Widget $i$, for $i = 1, 2, \ldots, 141$.

Let $y$ denote the amount (kg) of CatalystX sold (with $0 \leq y \leq 1500$).

Let $z$ denote the amount (kg) of CatalystX disposed (with $z \geq 0$).

##### Parameters

For each widget $i$ ($i = 1, \ldots, 141$):

- $l_i$: labor hours required per unit
- $a_i$: Material A required per unit (kg)
- $b_i$: Material B required per unit (kg)
- $p_i$: base profit per unit

From the data:

- $l_i, a_i, b_i, p_i$ are given for each Widget $i$ in the table below.
- Widget3 ($i=3$) generates 5 kg CatalystX per unit produced.

Resource limits:

- Total labor hours: $L_{max} = 5000$
- Total Material A: $A_{max} = 24000$
- Total Material B: $B_{max} = 15000$

CatalystX:

- Sale price: $300$ per kg, up to $1500$ kg per month
- Disposal cost: $200$ per kg for any unsold CatalystX

##### Objective Function

Maximize total profit:
\[
\max \left( \sum_{i=1}^{141} p_i x_i + 300y - 200z \right)
\]

##### Constraints

1. **Labor hours constraint:**
   \[
   \sum_{i=1}^{141} l_i x_i \leq 5000
   \]

2. **Material A constraint:**
   \[
   \sum_{i=1}^{141} a_i x_i \leq 24000
   \]

3. **Material B constraint:**
   \[
   \sum_{i=1}^{141} b_i x_i \leq 15000
   \]

4. **CatalystX balance:**
   \[
   5 x_3 = y + z
   \]

5. **CatalystX sales cap:**
   \[
   0 \leq y \leq 1500
   \]

6. **CatalystX disposal non-negativity:**
   \[
   z \geq 0
   \]

7. **Production non-negativity:**
   \[
   x_i \geq 0 \quad \forall i = 1, \ldots, 141
   \]

##### Retrieved Information

```json
{
  "resource_limits": {
    "LaborHours": 5000,
    "MaterialA": 24000,
    "MaterialB": 15000
  },
  "widgets": [
    {"Product": "Widget1", "LaborHours": 1.6, "MaterialA": 24, "MaterialB": 14, "Profit": 525},
    {"Product": "Widget2", "LaborHours": 2, "MaterialA": 20, "MaterialB": 10, "Profit": 678},
    {"Product": "Widget3", "LaborHours": 2.5, "MaterialA": 12, "MaterialB": 18, "Profit": 812},
    {"Product": "Widget4", "LaborHours": 1.9, "MaterialA": 21, "MaterialB": 15, "Profit": 769},
    {"Product": "Widget5", "LaborHours": 0.0, "MaterialA": 15, "MaterialB": 26, "Profit": 952},
    {"Product": "Widget6", "LaborHours": 0.1, "MaterialA": 24, "MaterialB": 17, "Profit": 987},
    {"Product": "Widget7", "LaborHours": 1.2, "MaterialA": 15, "MaterialB": 30, "Profit": 644},
    {"Product": "Widget8", "LaborHours": 1.3, "MaterialA": 21, "MaterialB": 24, "Profit": 795},
    {"Product": "Widget9", "LaborHours": 0.4, "MaterialA": 20, "MaterialB": 30, "Profit": 829},
    {"Product": "Widget10", "LaborHours": 0.9, "MaterialA": 18, "MaterialB": 27, "Profit": 574},
    {"Product": "Widget11", "LaborHours": 0.2, "MaterialA": 24, "MaterialB": 21, "Profit": 723},
    {"Product": "Widget12", "LaborHours": 1.6, "MaterialA": 20, "MaterialB": 20, "Profit": 775},
    {"Product": "Widget13", "LaborHours": 0.1, "MaterialA": 20, "MaterialB": 22, "Profit": 992},
    {"Product": "Widget14", "LaborHours": 0.7, "MaterialA": 12, "MaterialB": 19, "Profit": 568},
    {"Product": "Widget15", "LaborHours": 1.1, "MaterialA": 11, "MaterialB": 28, "Profit": 895},
    {"Product": "Widget16", "LaborHours": 0.7, "MaterialA": 15, "MaterialB": 25, "Profit": 838},
    {"Product": "Widget17", "LaborHours": 1.0, "MaterialA": 14, "MaterialB": 27, "Profit": 883},
    {"Product": "Widget18", "LaborHours": 1.8, "MaterialA": 18, "MaterialB": 15, "Profit": 567},
    {"Product": "Widget19", "LaborHours": 0.5, "MaterialA": 23, "MaterialB": 25, "Profit": 829},
    {"Product": "Widget20", "LaborHours": 1.2, "MaterialA": 17, "MaterialB": 29, "Profit": 821},
    {"Product": "Widget21", "LaborHours": 0.2, "MaterialA": 20, "MaterialB": 17, "Profit": 893},
    {"Product": "Widget22", "LaborHours": 1.5, "MaterialA": 10, "MaterialB": 22, "Profit": 977},
    {"Product": "Widget23", "LaborHours": 1.1, "MaterialA": 12, "MaterialB": 21, "Profit": 536},
    {"Product": "Widget24", "LaborHours": 0.6, "MaterialA": 12, "MaterialB": 24, "Profit": 797},
    {"Product": "Widget25", "LaborHours": 1.8, "MaterialA": 10, "MaterialB": 25, "Profit": 793},
    {"Product": "Widget26", "LaborHours": 2.0, "MaterialA": 15, "MaterialB": 23, "Profit": 754},
    {"Product": "Widget27", "LaborHours": 1.4, "MaterialA": 17, "MaterialB": 21, "Profit": 819},
    {"Product": "Widget28", "LaborHours": 1.3, "MaterialA": 22, "MaterialB": 30, "Profit": 947},
    {"Product": "Widget29", "LaborHours": 1.2, "MaterialA": 11, "MaterialB": 17, "Profit": 857},
    {"Product": "Widget30", "LaborHours": 0.3, "MaterialA": 13, "MaterialB": 29, "Profit": 886},
    {"Product": "Widget31", "LaborHours": 1.6, "MaterialA": 17, "MaterialB": 15, "Profit": 609},
    {"Product": "Widget32", "LaborHours": 0.9, "MaterialA": 24, "MaterialB": 19, "Profit": 706},
    {"Product": "Widget33", "LaborHours": 0.1, "MaterialA": 10, "MaterialB": 26, "Profit": 545},
    {"Product": "Widget34", "LaborHours": 0.0, "MaterialA": 12, "MaterialB": 22, "Profit": 762},
    {"Product": "Widget35", "LaborHours": 1.8, "MaterialA": 12, "MaterialB": 21, "Profit": 719},
    {"Product": "Widget36", "LaborHours": 0.3, "MaterialA": 24, "MaterialB": 26, "Profit": 793},
    {"Product": "Widget37", "LaborHours": 0.6, "MaterialA": 20, "MaterialB": 27, "Profit": 865},
    {"Product": "Widget38", "LaborHours": 0.7, "MaterialA": 16, "MaterialB": 23, "Profit": 932},
    {"Product": "Widget39", "LaborHours": 0.6, "MaterialA": 17, "MaterialB": 18, "Profit": 962},
    {"Product": "Widget40", "LaborHours": 1.3, "MaterialA": 10, "MaterialB": 22, "Profit": 824},
    {"Product": "Widget41", "LaborHours": 0.8, "MaterialA": 15, "MaterialB": 19, "Profit": 992},
    {"Product": "Widget42", "LaborHours": 1.7, "MaterialA": 21, "MaterialB": 29, "Profit": 536},
    {"Product": "Widget43", "LaborHours": 1.5, "MaterialA": 17, "MaterialB": 25, "Profit": 896},
    {"Product": "Widget44", "LaborHours": 1.3, "MaterialA": 25, "MaterialB": 23, "Profit": 969},
    {"Product": "Widget45", "LaborHours": 1.8, "MaterialA": 21, "MaterialB": 26, "Profit": 558},
    {"Product": "Widget46", "LaborHours": 0.1, "MaterialA": 25, "MaterialB": 20, "Profit": 893},
    {"Product": "Widget47", "LaborHours": 1.0, "MaterialA": 11, "MaterialB": 30, "Profit": 777},
    {"Product": "Widget48", "LaborHours": 0.6, "MaterialA": 23, "MaterialB": 29, "Profit": 718},
    {"Product": "Widget49", "LaborHours": 0.2, "MaterialA": 25, "MaterialB": 16, "Profit": 811},
    {"Product": "Widget50", "LaborHours": 1.2, "MaterialA": 25, "MaterialB": 29, "Profit": 545},
    {"Product": "Widget51", "LaborHours": 0.7, "MaterialA": 17, "MaterialB": 25, "Profit": 961},
    {"Product": "Widget52", "LaborHours": 1.7, "MaterialA": 12, "MaterialB": 25, "Profit": 657},
    {"Product": "Widget53", "LaborHours": 0.8, "MaterialA": 12, "MaterialB": 23, "Profit": 658},
    {"Product": "Widget54", "LaborHours": 1.6, "MaterialA": 15, "MaterialB": 19, "Profit": 860},
    {"Product": "Widget55", "LaborHours": 0.5, "MaterialA": 22, "MaterialB": 20, "Profit": 625},
    {"Product": "Widget56", "LaborHours": 1.1, "MaterialA": 18, "MaterialB": 17, "Profit": 510},
    {"Product": "Widget57", "LaborHours": 0.0, "MaterialA": 17, "MaterialB": 24, "Profit": 1000},
    {"Product": "Widget58", "LaborHours": 0.2, "MaterialA": 22, "MaterialB": 17, "Profit": 544},
    {"Product": "Widget59", "LaborHours": 1.7, "MaterialA": 23, "MaterialB": 17, "Profit": 547},
    {"Product": "Widget60", "LaborHours": 0.5, "MaterialA": 18, "MaterialB": 27, "Profit": 961},
    {"Product": "Widget61", "LaborHours": 1.6, "MaterialA": 16, "MaterialB": 23, "Profit": 689},
    {"Product": "Widget62", "LaborHours": 1.1, "MaterialA": 12, "MaterialB": 16, "Profit": 545},
    {"Product": "Widget63", "LaborHours": 0.5, "MaterialA": 13, "MaterialB": 16, "Profit": 897},
    {"Product": "Widget64", "LaborHours": 0.1, "MaterialA": 13, "MaterialB": 29, "Profit": 881},
    {"Product": "Widget65", "LaborHours": 0.6, "MaterialA": 25, "MaterialB": 24, "Profit": 972},
    {"Product": "Widget66", "LaborHours": 1.9, "MaterialA": 18, "MaterialB": 17, "Profit": 860},
    {"Product": "Widget67", "LaborHours": 0.7, "MaterialA": 23, "MaterialB": 15, "Profit": 881},
    {"Product": "Widget68", "LaborHours": 1.3, "MaterialA": 25, "MaterialB": 15, "Profit": 892},
    {"Product": "Widget69", "LaborHours": 1.5, "MaterialA": 22, "MaterialB": 27, "Profit": 797},
    {"Product": "Widget70", "LaborHours": 0.4, "MaterialA": 14, "MaterialB": 21, "Profit": 767},
    {"Product": "Widget71", "LaborHours": 1.1, "MaterialA": 14, "MaterialB": 22, "Profit": 794},
    {"Product": "Widget72", "LaborHours": 0.3, "MaterialA": 15, "MaterialB": 30, "Profit": 899},
    {"Product": "Widget73", "LaborHours": 0.6, "MaterialA": 10, "MaterialB": 26, "Profit": 504},
    {"Product": "Widget74", "LaborHours": 1.5, "MaterialA": 10, "MaterialB": 26, "Profit": 731},
    {"Product": "Widget75", "LaborHours": 0.2, "MaterialA": 23, "MaterialB": 19, "Profit": 929},
    {"Product": "Widget76", "LaborHours": 0.3, "MaterialA": 22, "MaterialB": 19, "Profit": 872},
    {"Product": "Widget77", "LaborHours": 1.0, "MaterialA": 14, "MaterialB": 25, "Profit": 631},
    {"Product": "Widget78", "LaborHours": 1.3, "MaterialA": 14, "MaterialB": 21, "Profit": 681},
    {"Product": "Widget79", "LaborHours": 2.0, "MaterialA": 22, "MaterialB": 28, "Profit": 587},
    {"Product": "Widget80", "LaborHours": 1.8, "MaterialA": 22, "MaterialB": 21, "Profit": 829},
    {"Product": "Widget81", "LaborHours": 1.2, "MaterialA": 12, "MaterialB": 29, "Profit": 734},
    {"Product": "Widget82", "LaborHours": 1.1, "MaterialA": 12, "MaterialB": 19, "Profit": 648},
    {"Product": "Widget83", "LaborHours": 0.3, "MaterialA": 22, "MaterialB": 26, "Profit": 738},
    {"Product": "Widget84", "LaborHours": 1.4, "MaterialA": 23, "MaterialB": 19, "Profit": 648},
    {"Product": "Widget85", "LaborHours": 1.1, "MaterialA": 11, "MaterialB": 30, "Profit": 557},
    {"Product": "Widget86", "LaborHours": 1.6, "MaterialA": 23, "MaterialB": 28, "Profit": 960},
    {"Product": "Widget87", "LaborHours": 1.6, "MaterialA": 14, "MaterialB": 15, "Profit": 907},
    {"Product": "Widget88", "LaborHours": 1.5, "MaterialA": 17, "MaterialB": 24, "Profit": 531},
    {"Product": "Widget89", "LaborHours": 0.0, "MaterialA": 23, "MaterialB": 21, "Profit": 887},
    {"Product": "Widget90", "LaborHours": 0.6, "MaterialA": 13, "MaterialB": 21, "Profit": 675},
    {"Product": "Widget91", "LaborHours": 0.2, "MaterialA": 25, "MaterialB": 20, "Profit": 766},
    {"Product": "Widget92", "LaborHours": 0.5, "MaterialA": 15, "MaterialB": 18, "Profit": 861},
    {"Product": "Widget93", "LaborHours": 1.4, "MaterialA": 24, "MaterialB": 20, "Profit": 841},
    {"Product": "Widget94", "LaborHours": 1.6, "MaterialA": 17, "MaterialB": 28, "Profit": 618},
    {"Product": "Widget95", "LaborHours": 1.8, "MaterialA": 23, "MaterialB": 17, "Profit": 574},
    {"Product": "Widget96", "LaborHours": 0.5, "MaterialA": 13, "MaterialB": 18, "Profit": 748},
    {"Product": "Widget97", "LaborHours": 1.6, "MaterialA": 20, "MaterialB": 21, "Profit": 898},
    {"Product": "Widget98", "LaborHours": 1.4, "MaterialA": 16, "MaterialB": 18, "Profit": 520},
    {"Product": "Widget99", "LaborHours": 0.3, "MaterialA": 11, "MaterialB": 19, "Profit": 675},
    {"Product": "Widget100", "LaborHours": 1.3, "MaterialA": 22, "MaterialB": 26, "Profit": 868},
    {"Product": "Widget101", "LaborHours": 1.5, "MaterialA": 18, "MaterialB": 19, "Profit": 698},
    {"Product": "Widget102", "LaborHours": 0.7, "MaterialA": 12, "MaterialB": 17, "Profit": 988},
    {"Product": "Widget103", "LaborHours": 0.8, "MaterialA": 12, "MaterialB": 29, "Profit": 550},
    {"Product": "Widget104", "LaborHours": 1.8, "MaterialA": 10, "MaterialB": 27, "Profit": 508},
    {"Product": "Widget105", "LaborHours": 1.7, "MaterialA": 24, "MaterialB": 22, "Profit": 682},
    {"Product": "Widget106", "LaborHours": 0.8, "MaterialA": 21, "MaterialB": 20, "Profit": 995},
    {"Product": "Widget107", "LaborHours": 0.3, "MaterialA": 25, "MaterialB": 30, "Profit": 656},
    {"Product": "Widget108", "LaborHours": 1.3, "MaterialA": 25, "MaterialB": 22, "Profit": 569},
    {"Product": "Widget109", "LaborHours": 0.9, "MaterialA": 21, "MaterialB": 18, "Profit": 779},
    {"Product": "Widget110", "LaborHours": 1.6, "MaterialA": 23, "MaterialB": 26, "Profit": 965},
    {"Product": "Widget111", "LaborHours": 0.4, "MaterialA": 23, "MaterialB": 27, "Profit": 675},
    {"Product": "Widget112", "LaborHours": 1.7, "MaterialA": 21, "MaterialB": 30, "Profit": 638},
    {"Product": "Widget113", "LaborHours": 1.5, "MaterialA": 20, "MaterialB": 26, "Profit": 888},
    {"Product": "Widget114", "LaborHours": 2.0, "MaterialA": 15, "MaterialB": 26, "Profit": 875},
    {"Product": "Widget115", "LaborHours": 0.9, "MaterialA": 24, "MaterialB": 24, "Profit": 552},
    {"Product": "Widget116", "LaborHours": 0.9, "MaterialA": 12, "MaterialB": 23, "Profit": 705},
    {"Product": "Widget117", "LaborHours": 0.1, "MaterialA": 15, "MaterialB": 28, "Profit": 863},
    {"Product": "Widget118", "LaborHours": 0.1, "MaterialA": 14, "MaterialB": 18, "Profit": 706},
    {"Product": "Widget119", "LaborHours": 0.3, "MaterialA": 13, "MaterialB": 25, "Profit": 663},
    {"Product": "Widget120", "LaborHours": 0.1, "MaterialA": 13, "MaterialB": 20, "Profit": 859},
    {"Product": "Widget121", "LaborHours": 1.3, "MaterialA": 24, "MaterialB": 17, "Profit": 809},
    {"Product": "Widget122", "LaborHours": 1.5, "MaterialA": 12, "MaterialB": 17, "Profit": 830},
    {"Product": "Widget123", "LaborHours": 1.7, "MaterialA": 19, "MaterialB": 29, "Profit": 991},
    {"Product": "Widget124", "LaborHours": 0.7, "MaterialA": 10, "MaterialB": 16, "Profit": 791},
    {"Product": "Widget125", "LaborHours": 0.6, "MaterialA": 13, "MaterialB": 22, "Profit": 934},
    {"Product": "Widget126", "LaborHours": 0.8, "MaterialA": 22, "MaterialB": 15, "Profit": 732},
    {"Product": "Widget127", "LaborHours": 0.4, "MaterialA": 22, "MaterialB": 17, "Profit": 989},
    {"Product": "Widget128", "LaborHours": 1.4, "MaterialA": 21, "MaterialB": 21, "Profit": 573},
    {"Product": "Widget129", "LaborHours": 0.1, "MaterialA": 16, "MaterialB": 23, "Profit": 720},
    {"Product": "Widget130", "LaborHours": 0.5, "MaterialA": 19, "MaterialB": 27, "Profit": 834},
    {"Product": "Widget131", "LaborHours": 0.1, "MaterialA": 23, "MaterialB": 18, "Profit": 577},
    {"Product": "Widget132", "LaborHours": 0.9, "MaterialA": 17, "MaterialB": 20, "Profit": 951},
    {"Product": "Widget133", "LaborHours": 1.1, "MaterialA": 19, "MaterialB": 30, "Profit": 882},
    {"Product": "Widget134", "LaborHours": 1.7, "MaterialA": 17, "MaterialB": 17, "Profit": 859},
    {"Product": "Widget135", "LaborHours": 1.3, "MaterialA": 22, "MaterialB": 17, "Profit": 568},
    {"Product": "Widget136", "LaborHours": 0.8, "MaterialA": 17, "MaterialB": 26, "Profit": 861},
    {"Product": "Widget137", "LaborHours": 1.7, "MaterialA": 13, "MaterialB": 17, "Profit": 875},
    {"Product": "Widget138", "LaborHours": 1.7, "MaterialA": 12, "MaterialB": 18, "Profit": 834},
    {"Product": "Widget139", "LaborHours": 1.6, "MaterialA": 12, "MaterialB": 23, "Profit": 836},
    {"Product": "Widget140", "LaborHours": 1.6, "MaterialA": 17, "MaterialB": 24, "Profit": 938},
    {"Product": "Widget141", "LaborHours": 1.2, "MaterialA": 11, "MaterialB": 16, "Profit": 593}
  ],
  "catalystX": {
    "byproduct_widget": "Widget3",
    "rate_per_unit": 5,
    "sale_price_per_kg": 300,
    "sale_cap_kg": 1500,
    "disposal_cost_per_kg": 200
  }
}
```

##### Variable Domains

\[
x_i \geq 0 \quad \forall i = 1, \ldots, 141
\]
\[
0 \leq y \leq 1500
\]
\[
z \geq 0
\]

##### Summary Table of Widget Parameters

| $i$ | Product    | $l_i$ | $a_i$ | $b_i$ | $p_i$ |
|-----|------------|-------|-------|-------|-------|
| 1   | Widget1    | 1.6   | 24    | 14    | 525   |
| 2   | Widget2    | 2     | 20    | 10    | 678   |
| 3   | Widget3    | 2.5   | 12    | 18    | 812   |
| ... | ...        | ...   | ...   | ...   | ...   |
| 141 | Widget141  | 1.2   | 11    | 16    | 593   |

(Full parameter list as above.)

##### Complete Mathematical Model

\[
\begin{align*}
\max \quad & \sum_{i=1}^{141} p_i x_i + 300y - 200z \\
\text{s.t.} \quad
& \sum_{i=1}^{141} l_i x_i \leq 5000 \\
& \sum_{i=1}^{141} a_i x_i \leq 24000 \\
& \sum_{i=1}^{141} b_i x_i \leq 15000 \\
& 5 x_3 = y + z \\
& 0 \leq y \leq 1500 \\
& z \geq 0 \\
& x_i \geq 0 \quad \forall i = 1, \ldots, 141
\end{align*}
\]

All coefficients and limits are as retrieved above.