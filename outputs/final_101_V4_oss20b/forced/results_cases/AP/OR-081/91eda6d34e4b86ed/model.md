**Mathematical Model**

Let  

- \(F\) be the set of all foods listed in *cost.csv*  
- For each \(f\in F\)  
  - \(c_f\)  – cost per serving (USD)  
  - \(cal_f\) – calories per serving (kcal)  
  - \(prot_f\) – protein per serving (g)  
  - \(fat_f\) – fat per serving (g)  
  - \(vitC_f\) – vitamin C per serving (mg)  

Decision variable  

- \(x_f \ge 0\) – number of servings of food \(f\) (fractional allowed)

\[
\begin{aligned}
\min\;& \sum_{f\in F} c_f\,x_f \\[4pt]
\text{s.t.}\;& \sum_{f\in F} cal_f\,x_f \;\ge\; 2000 \\[2pt]
& \sum_{f\in F} prot_f\,x_f \;\ge\; 50 \\[2pt]
& \sum_{f\in F} vitC_f\,x_f \;\ge\; 60 \\[2pt]
& \sum_{f\in F} fat_f\,x_f \;\le\; 70 \\[2pt]
& x_f \;\ge\; 0 \quad \forall f\in F
\end{aligned}
\]

---

**Data Mapping**

```json
{
  "table_id": "file_0_view_0",
  "columns": {
    "Food": "identifier",
    "Calories": "cal_f",
    "Protein(g)": "prot_f",
    "Fat(g)": "fat_f",
    "VitaminC(mg)": "vitC_f",
    "Cost": "c_f"
  },
  "index_set": "F",
  "parameter_mappings": {
    "c_f": {"source_column": "Cost"},
    "cal_f": {"source_column": "Calories"},
    "prot_f": {"source_column": "Protein(g)"},
    "fat_f": {"source_column": "Fat(g)"},
    "vitC_f": {"source_column": "VitaminC(mg)"}
  }
}
```

The mapping links each symbolic parameter to its corresponding column in the CSV file, and the set \(F\) is derived from all distinct values in the *Food* column. No numerical values are enumerated; the model is ready for a solver to ingest the full dataset.