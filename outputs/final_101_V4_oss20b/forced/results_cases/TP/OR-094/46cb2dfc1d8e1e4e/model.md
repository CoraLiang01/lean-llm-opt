##### Decision Variables

$x_k\in\mathbb{Z}_{\ge0}$: number of units of radio model HiFi-$k$, $k=1,\dots,101$.

##### Objective Function

$\displaystyle \min \sum_{i=1}^{3}\Bigl(C_i-\sum_{k=1}^{101}t_{ik}\,x_k\Bigr)$

where $C_i$ is the effective daily capacity of workstation $i$ and $t_{ik}$ is the processing time (minutes) required for one unit of HiFi-$k$ at workstation $i$.

##### Constraints

1. **Workstation capacity**  
   $\displaystyle \sum_{k=1}^{101}t_{ik}\,x_k\le C_i,\qquad i=1,2,3.$  

2. **Non‑negativity and integrality**  
   $\displaystyle x_k\in\mathbb{Z}_{\ge0},\qquad k=1,\dots,101.$  

##### Parameters

$C_1=1440\,(1-0.10)=1296$  
$C_2=1440\,(1-0.14)=1238.4$  
$C_3=1440\,(1-0.12)=1267.2$

$ t_{ik}$ (processing times in minutes):

| $k$ | $t_{1k}$ | $t_{2k}$ | $t_{3k}$ |
|-----|-----------|-----------|-----------|
| 1 | 6 | 5 | 4 |
| 2 | 4 | 5 | 6 |
| 3 | 6 | 5 | 5 |
| 4 | 7 | 1 | 2 |
| 5 | 6 | 7 | 6 |
| 6 | 6 | 8 | 5 |
| 7 | 8 | 7 | 3 |
| 8 | 9 | 5 | 3 |
| 9 | 6 | 6 | 4 |
|10 | 7 | 8 | 8 |
|11 | 1 | 9 | 6 |
|12 | 2 | 9 | 3 |
|13 | 4 | 2 | 3 |
|14 | 7 | 6 | 3 |
|15 | 3 | 9 | 7 |
|16 | 8 | 4 | 8 |
|17 | 3 | 1 | 3 |
|18 | 2 | 2 | 8 |
|19 | 4 | 9 | 1 |
|20 | 5 | 3 | 5 |
|21 | 8 | 8 | 3 |
|22 | 3 | 5 | 8 |
|23 | 2 | 9 | 5 |
|24 | 3 | 5 | 8 |
|25 | 9 | 8 | 4 |
|26 | 7 | 7 | 8 |
|27 | 3 | 1 | 6 |
|28 | 5 | 1 | 7 |
|29 | 7 | 9 | 9 |
|30 | 6 | 7 | 5 |
|31 | 2 | 1 | 3 |
|32 | 1 | 9 | 6 |
|33 | 5 | 6 | 3 |
|34 | 6 | 4 | 3 |
|35 | 5 | 7 | 3 |
|36 | 1 | 4 | 8 |
|37 | 7 | 8 | 4 |
|38 | 9 | 6 | 6 |
|39 | 8 | 5 | 3 |
|40 | 3 | 3 | 8 |
|41 | 3 | 6 | 3 |
|42 | 8 | 7 | 7 |
|43 | 2 | 6 | 5 |
|44 | 3 | 2 | 3 |
|45 | 3 | 1 | 1 |
|46 | 8 | 1 | 8 |
|47 | 9 | 3 | 9 |
|48 | 2 | 8 | 6 |
|49 | 3 | 4 | 6 |
|50 | 4 | 3 | 4 |
|51 | 2 | 6 | 7 |
|52 | 9 | 9 | 1 |
|53 | 2 | 8 | 9 |
|54 | 1 | 7 | 9 |
|55 | 8 | 2 | 3 |
|56 | 8 | 2 | 9 |
|57 | 4 | 5 | 6 |
|58 | 4 | 4 | 5 |
|59 | 6 | 3 | 7 |
|60 | 1 | 8 | 8 |
|61 | 6 | 8 | 9 |
|62 | 5 | 6 | 9 |
|63 | 3 | 6 | 8 |
|64 | 5 | 3 | 5 |
|65 | 1 | 1 | 4 |
|66 | 6 | 6 | 4 |
|67 | 6 | 2 | 3 |
|68 | 5 | 6 | 3 |
|69 | 3 | 1 | 8 |
|70 | 4 | 3 | 8 |
|71 | 3 | 7 | 2 |
|72 | 8 | 1 | 4 |
|73 | 1 | 1 | 9 |
|74 | 2 | 2 | 6 |
|75 | 3 | 8 | 7 |
|76 | 2 | 7 | 6 |
|77 | 8 | 8 | 7 |
|78 | 4 | 8 | 3 |
|79 | 4 | 7 | 1 |
|80 | 2 | 5 | 7 |
|81 | 7 | 2 | 6 |
|82 | 5 | 5 | 4 |
|83 | 1 | 6 | 3 |
|84 | 6 | 2 | 5 |
|85 | 4 | 3 | 7 |
|86 | 1 | 2 | 6 |
|87 | 3 | 3 | 3 |
|88 | 8 | 8 | 5 |
|89 | 3 | 4 | 2 |
|90 | 3 | 9 | 2 |
|91 | 3 | 6 | 9 |
|92 | 3 | 1 | 3 |
|93 | 6 | 4 | 6 |
|94 | 7 | 8 | 9 |
|95 | 6 | 8 | 7 |
|96 | 2 | 6 | 2 |
|97 | 1 | 8 | 4 |
|98 | 8 | 5 | 5 |
|99 | 9 | 5 | 8 |
|100 | 7 | 8 | 1 |
|101 | 9 | 3 | 6 |

##### Retrieved Information

```json
{
  "workstations": {
    "1": {
      "maintenance_percent": 10,
      "effective_capacity": 1296,
      "times": {
        "HiFi1": 6, "HiFi2": 4, "HiFi3": 6, "HiFi4": 7, "HiFi5": 6, "HiFi6": 6, "HiFi7": 8, "HiFi8": 9, "HiFi9": 6, "HiFi10": 7, "HiFi11": 1, "HiFi12": 2, "HiFi13": 4, "HiFi14": 7, "HiFi15": 3, "HiFi16": 8, "HiFi17": 3, "HiFi18": 2, "HiFi19": 4, "HiFi20": 5, "HiFi21": 8, "HiFi22": 3, "HiFi23": 2, "HiFi24": 3, "HiFi25": 9, "HiFi26": 7, "HiFi27": 3, "HiFi28": 5, "HiFi29": 7, "HiFi30": 6, "HiFi31": 2, "HiFi32": 1, "HiFi33": 5, "HiFi34": 6, "HiFi35": 5, "HiFi36": 1, "HiFi37": 7, "HiFi38": 9, "HiFi39": 8, "HiFi40": 3, "HiFi41": 3, "HiFi42": 8, "HiFi43": 2, "HiFi44": 3, "HiFi45": 3, "HiFi46": 8, "HiFi47": 9, "HiFi48": 2, "HiFi49": 3, "HiFi50": 4, "HiFi51": 2, "HiFi52": 9, "HiFi53": 2, "HiFi54": 1, "HiFi55": 8, "HiFi56": 8, "HiFi57": 4, "HiFi58": 4, "HiFi59": 6, "HiFi60": 1, "HiFi61": 6, "HiFi62": 5, "HiFi63": 3, "HiFi64": 5, "HiFi65": 1, "HiFi66": 6, "HiFi67": 6, "HiFi68": 5, "HiFi69": 3, "HiFi70": 4, "HiFi71": 3, "HiFi72": 8, "HiFi73": 1, "HiFi74": 2, "HiFi75": 3, "HiFi76": 2, "HiFi77": 8, "HiFi78": 4, "HiFi79": 4, "HiFi80": 2, "HiFi81": 7, "HiFi82": 5, "HiFi83": 1, "HiFi84": 6, "HiFi85": 4, "HiFi86": 1, "HiFi87": 3, "HiFi88": 8, "HiFi89": 3, "HiFi90": 3, "HiFi91": 3, "HiFi92": 3, "HiFi93": 6, "HiFi94": 7, "HiFi95": 6, "HiFi96": 2, "HiFi97": 1, "HiFi98": 8, "HiFi99": 9, "HiFi100": 7, "HiFi101": 9
      }
    },
    "2": {
      "maintenance_percent": 14,
      "effective_capacity": 1238.4,
      "times": {
        "HiFi1": 5, "HiFi2": 5, "HiFi3": 5, "HiFi4": 1, "HiFi5": 7, "HiFi6": 8, "HiFi7": 7, "HiFi8": 5, "HiFi9": 6, "HiFi10": 8, "HiFi11": 9, "HiFi12": 9, "HiFi13": 2, "HiFi14": 6, "HiFi15": 9, "HiFi16": 4, "HiFi17": 1, "HiFi18": 2, "HiFi19": 9, "HiFi20": 3, "HiFi21": 8, "HiFi22": 5, "HiFi23": 9, "HiFi24": 5, "HiFi25": 8, "HiFi26": 7, "HiFi27": 1, "HiFi28": 1, "HiFi29": 9, "HiFi30": 7, "HiFi31": 1, "HiFi32": 9, "HiFi33": 6, "HiFi34": 4, "HiFi35": 7, "HiFi36": 4, "HiFi37": 8, "HiFi38": 6, "HiFi39": 5, "HiFi40": 3, "HiFi41": 6, "HiFi42": 7, "HiFi43": 6, "HiFi44": 2, "HiFi45": 1, "HiFi46": 1, "HiFi47": 3, "HiFi48": 8, "HiFi49": 4, "HiFi50": 3, "HiFi51": 6, "HiFi52": 9, "HiFi53": 8, "HiFi54": 7, "HiFi55": 2, "HiFi56": 2, "HiFi57": 5, "HiFi58": 4, "HiFi59": 3, "HiFi60": 8, "HiFi61": 8, "HiFi62": 6, "HiFi63": 6, "HiFi64": 3, "HiFi65": 1, "HiFi66": 6, "HiFi67": 2, "HiFi68": 6, "HiFi69": 1, "HiFi70": 3, "HiFi71": 7, "HiFi72": 1, "HiFi73": 1, "HiFi74": 2, "HiFi75": 8, "HiFi76": 7, "HiFi77": 8, "HiFi78": 8, "HiFi79": 7, "HiFi80": 5, "HiFi81": 2, "HiFi82": 5, "HiFi83": 6, "HiFi84": 2, "HiFi85": 3, "HiFi86": 2, "HiFi87": 3, "HiFi88": 8, "HiFi89": 4, "HiFi90": 9, "HiFi91": 6, "HiFi92": 1, "HiFi93": 4, "HiFi94": 8, "HiFi95": 8, "HiFi96": 6, "HiFi97": 8, "HiFi98": 5, "HiFi99": 5, "HiFi100": 8, "HiFi101": 3
      }
    },
    "3": {
      "maintenance_percent": 12,
      "effective_capacity": 1267.2,
      "times": {
        "HiFi1": 4, "HiFi2": 6, "HiFi3": 5, "HiFi4": 2, "HiFi5": 6, "HiFi6": 5, "HiFi7": 3, "HiFi8": 3, "HiFi9": 4, "HiFi10": 8, "HiFi11": 6, "HiFi12": 3, "HiFi13": 3, "HiFi14": 3, "HiFi15": 7, "HiFi16": 8, "HiFi17": 3, "HiFi18": 8, "HiFi19": 1, "HiFi20": 5, "HiFi21": 3, "HiFi22": 8, "HiFi23": 5, "HiFi24": 8, "HiFi25": 4, "HiFi26": 8, "HiFi27": 6, "HiFi28": 7, "HiFi29": 9, "HiFi30": 5, "HiFi31": 3, "HiFi32": 6, "HiFi33": 3, "HiFi34": 3, "HiFi35": 3, "HiFi36": 8, "HiFi37": 4, "HiFi38": 6, "HiFi39": 3, "HiFi40": 8, "HiFi41": 3, "HiFi42": 7, "HiFi43": 5, "HiFi44": 3, "HiFi45": 1, "HiFi46": 8, "HiFi47": 9, "HiFi48": 6, "HiFi49": 6, "HiFi50": 4, "HiFi51": 7, "HiFi52": 1, "HiFi53": 9, "HiFi54": 9, "HiFi55": 3, "HiFi56": 9, "HiFi57": 6, "HiFi58": 5, "HiFi59": 7, "HiFi60": 8, "HiFi61": 9, "HiFi62": 9, "HiFi63": 8, "HiFi64": 5, "HiFi65": 4, "HiFi66": 4, "HiFi67": 3, "HiFi68": 3, "HiFi69": 8, "HiFi70": 8, "HiFi71": 2, "HiFi72": 4, "HiFi73": 9, "HiFi74": 6, "HiFi75": 7, "HiFi76": 6, "HiFi77": 7, "HiFi78": 3, "HiFi79": 1, "HiFi80": 7, "HiFi81": 6, "HiFi82": 4, "HiFi83": 3, "HiFi84": 5, "HiFi85": 7, "HiFi86": 6, "HiFi87": 3, "HiFi88": 5, "HiFi89": 2, "HiFi90": 2, "HiFi91": 9, "HiFi92": 3, "HiFi93": 6, "HiFi94": 9, "HiFi95": 7, "HiFi96": 2, "HiFi97": 4, "HiFi98": 5, "HiFi99": 8, "HiFi100": 1, "HiFi101": 6
      }
    }
  }
}
```