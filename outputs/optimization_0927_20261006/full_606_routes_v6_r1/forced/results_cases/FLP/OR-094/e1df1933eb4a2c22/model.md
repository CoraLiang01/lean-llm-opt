##### Sets and Indices

- Let $M = \{1,2,\ldots,101\}$ be the set of radio models, indexed by $m$ (corresponding to HiFi1, HiFi2, ..., HiFi101).
- Let $W = \{1,2,3\}$ be the set of workstations, indexed by $w$.

##### Parameters

- $t_{w,m}$: Processing time (in minutes) required at workstation $w$ for one unit of model $m$.
- $C_w$: Effective daily capacity (in minutes) of workstation $w$ after maintenance.

From the data:

- Total available time per workstation: $1440$ minutes.
- Maintenance percentages:
  - Workstation 1: $10\%$ $\implies$ $C_1 = 1440 \times 0.90 = 1296$ minutes.
  - Workstation 2: $14\%$ $\implies$ $C_2 = 1440 \times 0.86 = 1238.4$ minutes.
  - Workstation 3: $12\%$ $\implies$ $C_3 = 1440 \times 0.88 = 1267.2$ minutes.

- $t_{w,m}$ values are given in the CSV as follows (partial for illustration; full data included):

For $w=1$ (Workstation 1):

| Model         | HiFi1 | HiFi2 | HiFi3 | HiFi4 | HiFi5 | ... | HiFi101 |
|---------------|-------|-------|-------|-------|-------|-----|---------|
| $t_{1,m}$     | 6     | 4     | 6     | 7     | 6     | ... | 9       |

For $w=2$ (Workstation 2):

| Model         | HiFi1 | HiFi2 | HiFi3 | HiFi4 | HiFi5 | ... | HiFi101 |
|---------------|-------|-------|-------|-------|-------|-----|---------|
| $t_{2,m}$     | 5     | 5     | 5     | 1     | 7     | ... | 3       |

For $w=3$ (Workstation 3):

| Model         | HiFi1 | HiFi2 | HiFi3 | HiFi4 | HiFi5 | ... | HiFi101 |
|---------------|-------|-------|-------|-------|-------|-----|---------|
| $t_{3,m}$     | 4     | 6     | 5     | 2     | 6     | ... | 6       |

(See below for the full $t_{w,m}$ matrix.)

##### Decision Variables

- $x_m \in \mathbb{Z}_{\geq 0}$: Number of units of model $m$ to produce per day (integer, nonnegative).
- $I_w \geq 0$: Idle time (in minutes) at workstation $w$ (continuous, nonnegative).

##### Objective Function

Minimize total idle time across all workstations:
\[
\min \sum_{w=1}^3 I_w
\]

##### Constraints

1. Idle time definition for each workstation:
   \[
   I_w = C_w - \sum_{m=1}^{101} t_{w,m} x_m, \quad \forall w \in \{1,2,3\}
   \]
2. Idle time cannot be negative:
   \[
   I_w \geq 0, \quad \forall w \in \{1,2,3\}
   \]
3. Nonnegativity and integrality of production:
   \[
   x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in \{1,2,\ldots,101\}
   \]

##### Full Parameter Data

- $C_1 = 1296$, $C_2 = 1238.4$, $C_3 = 1267.2$
- $t_{w,m}$ matrix (from workstation_times.csv):

Workstation 1 ($w=1$):

| Model         | HiFi1 | HiFi2 | HiFi3 | HiFi4 | HiFi5 | HiFi6 | HiFi7 | HiFi8 | HiFi9 | HiFi10 | ... | HiFi101 |
|---------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|-----|---------|
| $t_{1,m}$     | 6     | 4     | 6     | 7     | 6     | 6     | 8     | 9     | 6     | 7      | ... | 9       |

Workstation 2 ($w=2$):

| Model         | HiFi1 | HiFi2 | HiFi3 | HiFi4 | HiFi5 | HiFi6 | HiFi7 | HiFi8 | HiFi9 | HiFi10 | ... | HiFi101 |
|---------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|-----|---------|
| $t_{2,m}$     | 5     | 5     | 5     | 1     | 7     | 8     | 7     | 5     | 6     | 8      | ... | 3       |

Workstation 3 ($w=3$):

| Model         | HiFi1 | HiFi2 | HiFi3 | HiFi4 | HiFi5 | HiFi6 | HiFi7 | HiFi8 | HiFi9 | HiFi10 | ... | HiFi101 |
|---------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|-----|---------|
| $t_{3,m}$     | 4     | 6     | 5     | 2     | 6     | 5     | 3     | 3     | 4     | 8      | ... | 6       |

(Full $t_{w,m}$ values for all $w$ and $m$ are as retrieved above.)

##### Complete Mathematical Model

\[
\begin{align*}
\min \quad & I_1 + I_2 + I_3 \\
\text{s.t.} \quad
& I_1 = 1296 - \sum_{m=1}^{101} t_{1,m} x_m \\
& I_2 = 1238.4 - \sum_{m=1}^{101} t_{2,m} x_m \\
& I_3 = 1267.2 - \sum_{m=1}^{101} t_{3,m} x_m \\
& I_w \geq 0 \quad \forall w \in \{1,2,3\} \\
& x_m \in \mathbb{Z}_{\geq 0} \quad \forall m \in \{1,2,\ldots,101\}
\end{align*}
\]

where $t_{w,m}$ are as given in workstation_times.csv for all $w$ and $m$.

##### Retrieved Information

{
  "workstations": [
    {
      "Workstation": "1",
      "Maintenance_Percent": "10",
      "C": 1296,
      "t_wm": {
        "HiFi1": 6, "HiFi2": 4, "HiFi3": 6, "HiFi4": 7, "HiFi5": 6, "HiFi6": 6, "HiFi7": 8, "HiFi8": 9, "HiFi9": 6, "HiFi10": 7, "HiFi11": 1, "HiFi12": 2, "HiFi13": 4, "HiFi14": 7, "HiFi15": 3, "HiFi16": 8, "HiFi17": 3, "HiFi18": 2, "HiFi19": 4, "HiFi20": 5, "HiFi21": 8, "HiFi22": 3, "HiFi23": 2, "HiFi24": 3, "HiFi25": 9, "HiFi26": 7, "HiFi27": 3, "HiFi28": 5, "HiFi29": 7, "HiFi30": 6, "HiFi31": 2, "HiFi32": 1, "HiFi33": 5, "HiFi34": 6, "HiFi35": 5, "HiFi36": 1, "HiFi37": 7, "HiFi38": 9, "HiFi39": 8, "HiFi40": 3, "HiFi41": 3, "HiFi42": 8, "HiFi43": 2, "HiFi44": 3, "HiFi45": 3, "HiFi46": 8, "HiFi47": 9, "HiFi48": 2, "HiFi49": 3, "HiFi50": 4, "HiFi51": 2, "HiFi52": 9, "HiFi53": 2, "HiFi54": 1, "HiFi55": 8, "HiFi56": 8, "HiFi57": 4, "HiFi58": 4, "HiFi59": 6, "HiFi60": 1, "HiFi61": 6, "HiFi62": 5, "HiFi63": 3, "HiFi64": 5, "HiFi65": 1, "HiFi66": 6, "HiFi67": 6, "HiFi68": 5, "HiFi69": 3, "HiFi70": 4, "HiFi71": 3, "HiFi72": 8, "HiFi73": 1, "HiFi74": 2, "HiFi75": 3, "HiFi76": 2, "HiFi77": 8, "HiFi78": 4, "HiFi79": 4, "HiFi80": 2, "HiFi81": 7, "HiFi82": 5, "HiFi83": 1, "HiFi84": 6, "HiFi85": 4, "HiFi86": 1, "HiFi87": 3, "HiFi88": 8, "HiFi89": 3, "HiFi90": 3, "HiFi91": 3, "HiFi92": 3, "HiFi93": 6, "HiFi94": 7, "HiFi95": 6, "HiFi96": 2, "HiFi97": 1, "HiFi98": 8, "HiFi99": 9, "HiFi100": 7, "HiFi101": 9
      }
    },
    {
      "Workstation": "2",
      "Maintenance_Percent": "14",
      "C": 1238.4,
      "t_wm": {
        "HiFi1": 5, "HiFi2": 5, "HiFi3": 5, "HiFi4": 1, "HiFi5": 7, "HiFi6": 8, "HiFi7": 7, "HiFi8": 5, "HiFi9": 6, "HiFi10": 8, "HiFi11": 9, "HiFi12": 9, "HiFi13": 2, "HiFi14": 6, "HiFi15": 9, "HiFi16": 4, "HiFi17": 1, "HiFi18": 2, "HiFi19": 9, "HiFi20": 3, "HiFi21": 8, "HiFi22": 5, "HiFi23": 9, "HiFi24": 5, "HiFi25": 8, "HiFi26": 7, "HiFi27": 1, "HiFi28": 1, "HiFi29": 9, "HiFi30": 7, "HiFi31": 1, "HiFi32": 9, "HiFi33": 6, "HiFi34": 4, "HiFi35": 7, "HiFi36": 4, "HiFi37": 8, "HiFi38": 6, "HiFi39": 5, "HiFi40": 3, "HiFi41": 6, "HiFi42": 7, "HiFi43": 6, "HiFi44": 2, "HiFi45": 1, "HiFi46": 1, "HiFi47": 3, "HiFi48": 8, "HiFi49": 4, "HiFi50": 3, "HiFi51": 6, "HiFi52": 9, "HiFi53": 8, "HiFi54": 7, "HiFi55": 2, "HiFi56": 2, "HiFi57": 5, "HiFi58": 4, "HiFi59": 3, "HiFi60": 8, "HiFi61": 8, "HiFi62": 6, "HiFi63": 6, "HiFi64": 3, "HiFi65": 1, "HiFi66": 6, "HiFi67": 2, "HiFi68": 6, "HiFi69": 1, "HiFi70": 3, "HiFi71": 7, "HiFi72": 1, "HiFi73": 1, "HiFi74": 2, "HiFi75": 8, "HiFi76": 7, "HiFi77": 8, "HiFi78": 8, "HiFi79": 7, "HiFi80": 5, "HiFi81": 2, "HiFi82": 5, "HiFi83": 6, "HiFi84": 2, "HiFi85": 3, "HiFi86": 2, "HiFi87": 3, "HiFi88": 8, "HiFi89": 4, "HiFi90": 9, "HiFi91": 6, "HiFi92": 1, "HiFi93": 4, "HiFi94": 8, "HiFi95": 8, "HiFi96": 6, "HiFi97": 8, "HiFi98": 5, "HiFi99": 5, "HiFi100": 8, "HiFi101": 3
      }
    },
    {
      "Workstation": "3",
      "Maintenance_Percent": "12",
      "C": 1267.2,
      "t_wm": {
        "HiFi1": 4, "HiFi2": 6, "HiFi3": 5, "HiFi4": 2, "HiFi5": 6, "HiFi6": 5, "HiFi7": 3, "HiFi8": 3, "HiFi9": 4, "HiFi10": 8, "HiFi11": 6, "HiFi12": 3, "HiFi13": 3, "HiFi14": 3, "HiFi15": 7, "HiFi16": 8, "HiFi17": 3, "HiFi18": 8, "HiFi19": 1, "HiFi20": 5, "HiFi21": 3, "HiFi22": 8, "HiFi23": 5, "HiFi24": 8, "HiFi25": 4, "HiFi26": 8, "HiFi27": 6, "HiFi28": 7, "HiFi29": 9, "HiFi30": 5, "HiFi31": 3, "HiFi32": 6, "HiFi33": 3, "HiFi34": 3, "HiFi35": 3, "HiFi36": 8, "HiFi37": 4, "HiFi38": 6, "HiFi39": 3, "HiFi40": 8, "HiFi41": 3, "HiFi42": 7, "HiFi43": 5, "HiFi44": 3, "HiFi45": 1, "HiFi46": 8, "HiFi47": 9, "HiFi48": 6, "HiFi49": 6, "HiFi50": 4, "HiFi51": 7, "HiFi52": 1, "HiFi53": 9, "HiFi54": 9, "HiFi55": 3, "HiFi56": 9, "HiFi57": 6, "HiFi58": 5, "HiFi59": 7, "HiFi60": 8, "HiFi61": 9, "HiFi62": 9, "HiFi63": 8, "HiFi64": 5, "HiFi65": 4, "HiFi66": 4, "HiFi67": 3, "HiFi68": 3, "HiFi69": 8, "HiFi70": 8, "HiFi71": 2, "HiFi72": 4, "HiFi73": 9, "HiFi74": 6, "HiFi75": 7, "HiFi76": 6, "HiFi77": 7, "HiFi78": 3, "HiFi79": 1, "HiFi80": 7, "HiFi81": 6, "HiFi82": 4, "HiFi83": 3, "HiFi84": 5, "HiFi85": 7, "HiFi86": 6, "HiFi87": 3, "HiFi88": 5, "HiFi89": 2, "HiFi90": 2, "HiFi91": 9, "HiFi92": 3, "HiFi93": 6, "HiFi94": 9, "HiFi95": 7, "HiFi96": 2, "HiFi97": 4, "HiFi98": 5, "HiFi99": 8, "HiFi100": 1, "HiFi101": 6
      }
    }
  ]
}

All $t_{w,m}$ values are as above for $w=1,2,3$ and $m=1,\ldots,101$ (HiFi1, ..., HiFi101).