**

The program now reads the prepared JSON file, extracts the four tables by their names, converts the numeric columns to floats, and builds a fixed‑charge transportation model in Gurobi.  
Missing cost entries are handled by assigning a default value of `0.0`.  
The model is exposed as the variable `model` and is left active for inspection.