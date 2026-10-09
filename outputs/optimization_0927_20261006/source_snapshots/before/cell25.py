def build_few_shot_Other(examples, t="Model"):
    """Render already-structured examples; inserted prompt values need no brace escaping."""
    field = "Label" if t == "Model" else "Code"
    thought = (
        "I need to create an Abstract Model Plan based on the query and CSV schema."
        if t == "Model" else
        "I need to generate a single, complete, executable Gurobi Python code block."
    )
    return "\n\n".join(
        f"<EXAMPLE>\nQuestion: {example['prompt']}\nThought: {thought}\n"
        f"Final Answer:\n{example[field]}\n</EXAMPLE>"
        for example in examples if example[field].strip()
    )


def csv_schema_preview(dataset_address, query=""):
    """Preview plus full-file evidence; matching never chooses or filters the model data."""
    normalize = lambda value: re.sub(r"\s+", " ", str(value)).strip().casefold()
    normalized_query = normalize(query)
    quoted = re.findall(r'"([^"\n]+)"|“([^”\n]+)”|‘([^’\n]+)’|(?<!\w)\x27([^\x27\n]+)\x27(?!\w)', query)
    names = {normalize(next(part for part in group if part)) for group in quoted}
    blocks = []
    for raw in normalize_data_address(dataset_address).splitlines():
        path = Path(raw).expanduser()
        path = path if path.is_absolute() else PROJECT_ROOT / path
        typed = read_csv_compat(path)
        frame = read_csv_compat(path, dtype=str, keep_default_na=False)
        stats, columns, terms = {}, {}, set(names)
        for column in frame:
            values = frame[column]
            normalized = values.map(normalize)
            columns[column] = normalized
            present = values[normalized.ne("")]
            numbers = pd.to_numeric(present, errors="coerce")
            stats[column] = {"missing": int(normalized.eq("").sum()), "unique_nonempty": int(present.nunique())}
            if len(present) and numbers.notna().all():
                stats[column]["numeric_range"] = [float(numbers.min()), float(numbers.max())]
            # Unquoted evidence must literally name a real column or complete nonnumeric value.
            candidates = [column] + present.unique().tolist()
            for value in candidates:
                term = normalize(value)
                if term and not re.fullmatch(r"[-+\d.,%]+", term) and re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", normalized_query):
                    terms.add(term)
        evidence = []
        for term in sorted(terms):
            matches = []
            for column, values in columns.items():
                exact, prefix, contains = values.eq(term), values.str.startswith(term), values.str.contains(term, regex=False)
                if contains.any():
                    matches.append({"column": column, "exact": int(exact.sum()), "prefix": int(prefix.sum()),
                                    "contains": int(contains.sum()), "examples": frame.loc[contains, column].drop_duplicates().head(3).tolist()})
            evidence.append({"term": term, "matching_columns": matches,
                             "exact_matching_columns": sum(item["exact"] > 0 for item in matches)})
        blocks.append(
            f"File: {path.resolve()}\nCSV delimiter: comma; first line consumed as header.\nTotal rows: {len(frame)}\nColumns: {list(frame.columns)}\n"
            f"Parsed column types: {typed.dtypes.astype(str).to_dict()}\n"
            f"Preview only (first 10 rows):\n{frame.head(10).to_string(index=False)}\n"
            f"Full-file column statistics: {json.dumps(stats, ensure_ascii=False)}\n"
            "Matching uses casefold and collapsed/trimmed whitespace; original IDs and leading zeros are preserved.\n"
            "Counts are per column and per individual condition, not intersections. Empty matching_columns means zero matches in every column.\n"
            f"Query-name evidence: {json.dumps(evidence, ensure_ascii=False)}"
        )
    if not blocks:
        raise ValueError("No external CSV paths were supplied")
    return "\n\n---\n\n".join(blocks)


def get_Others_response(user_query: str, dataset_address: str):
    examples = retrieve_rag_examples("Others", user_query, k=3)
    llm2 = make_llm()

    try:
        print("\n[Gurobi Pipeline Step 1/3]: Getting CSV Scheme...")

        schema = csv_schema_preview(dataset_address, query=user_query)
        if re.search(r"\bsymmetric\b", user_query, re.I):
            schema += ("\nWhen the query explicitly states symmetry, align matrix axes by entity ID. "
                       "Fill a missing off-diagonal entry from its present transpose; preserve zeros. "
                       "Reject pairs with both entries missing or conflicting values beyond numeric tolerance. "
                       "Without explicit symmetry, preserve direction and never mirror or zero-fill entries.")
        if re.search(r"\bshut(?:down| down)\b", user_query, re.I):
            schema += "\nMinimum-down-time restrictions activate only on an on-to-off transition; down-time constraints alone must never forbid an always-on sequence."
        if "\\sum" in user_query:
            schema += "\nPreserve the query's explicit summation scope: aggregate every summed index inside the expression; never replace a summed index with a separate family of constraints."
        if re.search(r"\bworkers?\b", user_query, re.I) and re.search(r"\btasks?\b", user_query, re.I):
            schema += "\nIn a worker-task matrix, distinguish embedded row/column-axis captions from actual workers. Select workers by their identifiers, exclude label rows, then validate worker counts; never truncate by position."
        print("\n[Gurobi Pipeline Step 2/3]: Constructing Abstract Model...")

        few_shot_block_abstract = build_few_shot_Other(examples, t='Model')
        print(f"[Few-Shot Abstract Examples]:\n{few_shot_block_abstract}...")
        
        abstract_model_template = """
You are an expert optimization modeler.
Your task is to create an "Abstract Model Plan" based on the user's query and the CSV Schema (data structure).
This plan is *not* Gurobi code or mathematical formulas, but a clear, step-by-step reasoning process in English.

[Examples]
{few_shot_examples}

[Current Task]
User Query: {query}

CSV Schema:
{schema}

An explicitly enumerated complete entity set in the query is authoritative: extra CSV entities
must not enlarge that set unless the query explicitly delegates its definition to the data.
For a one-off finite-horizon schedule, change/ramp constraints compare consecutive modeled periods. An initial
binary on/off state is not a pre-horizon continuous quantity: do not invent an initial continuous
level or a first-period ramp limit unless the query explicitly supplies that continuous boundary.
For recurring daily/cyclic operation, preserve wrap-around coverage across the last/first period
when a working shift spans the boundary; do not impose a fictitious empty start to a recurring day.
State and implement the original objective sense and complete value expression, including constants.
Do not offer sign-reversed or constant-shifted optimizer-equivalent objectives as alternatives.
Examples illustrate style; the current query and schema determine the actual scope, indices and boundaries.

Data selection: state for each entity table whether all rows or a subset is required; name the
filter column, exact condition, entity key, and joins. Preserve shared resource/parameter tables.
Without a query-requested filter, use the complete entity set. A preview is never the selected set.
Matching statistics are evidence, not instructions: do not broaden a zero-match condition to
substring matching or all rows. Check selected keys and applicable counts at runtime; recompute
combined conditions rather than treating individual counts as intersection counts.

[Your Output]
You must strictly follow this format for your "Abstract Model Plan" output:

[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to {{{{Your analysis}}}}.
2.  **Identify Model Type:** Based on the query, this is a {{{{e.g., LP, MIP, Fixed-Charge, Blending}}}} problem.
3.  **Define Index Sets:** The primary indices are {{{{e.g., Products, Workers, Sources, Destinations}}}}.
4.  **Define Decision Variables:**
    -   `x[i]` = {{{{Describe first variable, e.g., 'quantity of product i'}}}}. Type: {{{{GRB.CONTINUOUS / GRB.INTEGER}}}}.
    -   `y[i]` = {{{{Describe second variable, e.g., 'if product i is produced'}}}}. Type: {{{{GRB.BINARY}}}}.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients (e.g., profit) will come from column(s): {{{{e.g., 'Price', 'Production Cost'}}}}.
    -   Constraint coefficients (e.g., resource use) will come from: {{{{e.g., 'Resource 1', 'Available Time'}}}}.
    -   Constraint RHS (limits) will come from: {{{{e.g., 'Max Demand'}}}}.
6.  **Formulate Objective:** {{{{Describe objective, e.g., 'Maximize sum((schema['Price'] - schema['Cost']) * x[i] - schema['FixedCost'] * y[i])'}}}}.
7.  **Formulate Constraints:**
    -   Constraint 1 (e.g., Resource Limit): {{{{Describe constraint 1, e.g., 'sum(schema['Resource 1'][i] * x[i]) <= schema['Available']'}}}}.
    -   Constraint 2 (e.g., Linking): {{{{Describe constraint 2, e.g., 'x[i] <= M * y[i])'}}}}.
    -   ... (Other constraints) ...
[Abstract Model Plan END]
"""
        abstract_prompt = PromptTemplate(
            template=abstract_model_template,
            input_variables=["query", "schema", "few_shot_examples"]
        )
        
        abstract_model_chain = LLMChain(llm=llm2, prompt=abstract_prompt)
        
        abstract_model_plan = abstract_model_chain.run(
            query=user_query,
            schema=schema,
            few_shot_examples=few_shot_block_abstract
        )
        print(f"[Observation]:\n{abstract_model_plan}")
        
        print("\n[Gurobi Pipeline Step 3/3]: Generating Gurobi Code...")
        
        few_shot_block_code = build_few_shot_Other(examples, t='Code')
        
        code_gen_template = """
You are an expert Gurobi programmer.
Your task is to strictly follow the User Query, CSV Schema, and "Abstract Model Plan" to translate them into a single, complete, executable Gurobi Python code block.
The code must start with ```python and end with ```.
The code must include the necessary imports from gurobipy, pandas, numpy, re, math or json. Do not import sys or os.
Create exactly one Gurobi model in m or model and call its optimize() exactly once, both at top level. Do not wrap model creation or solving in functions or try/except; let errors propagate to the execution harness. Do not write files or call computeIIS().
The code must read all supplied CSV files needed by the model, at these exact newline-separated paths:
{dataset_address}
Read their complete contents at runtime. The schema preview is not the full dataset.
Preserve explicit business identifiers and matrix axes; do not invent filenames or columns.
Data-reading contract: these files were parsed with pandas.read_csv(path, sep=',').
The displayed preview is space-aligned for readability; it does not imply a whitespace delimiter.
pandas already consumes the CSV header. Select entity and parameter rows through identifier values
or validated predicates, never guessed iloc offsets or a guessed number of header rows.
Normalize identifier types explicitly on both sides of every lookup/join, including matrix row
indices versus column labels. Preserve the actual identifiers and derive dimensions from all rows.
Series.apply receives one cell value, not a row; use vectorized column predicates or DataFrame.apply
with axis=1 when a predicate needs a row. Every comprehension must use the indices it binds, not
stale variables from a preceding loop. Do not silently skip failed conversions or missing coefficients.
Use setObjective for the exact metric requested by the query, including all additive constants,
signs and units. An optimizer-equivalent objective is not an equivalent reported objective value;
do not drop constant terms and then merely restore them in printed output.
The code must *fully* implement all variables, objectives, and constraints from the "Abstract Model Plan".

[Examples of the Full Process]
{few_shot_examples}

[User Query]
{query}

[CSV Schema]
{schema}

[Abstract Model Plan]
{abstract_plan}

An explicitly enumerated complete entity set in the query is authoritative: extra CSV entities
must not enlarge that set unless the query explicitly delegates its definition to the data.
For a one-off finite-horizon schedule, change/ramp constraints compare consecutive modeled periods. An initial
binary on/off state is not a pre-horizon continuous quantity: do not invent an initial continuous
level or a first-period ramp limit unless the query explicitly supplies that continuous boundary.
For recurring daily/cyclic operation, preserve wrap-around coverage across the last/first period
when a working shift spans the boundary; do not impose a fictitious empty start to a recurring day.
State and implement the original objective sense and complete value expression, including constants.
Do not offer sign-reversed or constant-shifted optimizer-equivalent objectives as alternatives.
Examples illustrate style; the current query and schema determine the actual scope, indices and boundaries.

Data selection: state for each entity table whether all rows or a subset is required; name the
filter column, exact condition, entity key, and joins. Preserve shared resource/parameter tables.
Without a query-requested filter, use the complete entity set. A preview is never the selected set.
Matching statistics are evidence, not instructions: do not broaden a zero-match condition to
substring matching or all rows. Check selected keys and applicable counts at runtime; recompute
combined conditions rather than treating individual counts as intersection counts.

Reserve m and model for the Gurobi model; never reuse them as loop variables or other values.
Keep business identifiers as Python dictionary keys, separate from Gurobi names.
Use name='' for addVars/addConstrs so business keys are not expanded into solver names;
use only short ASCII names for individually named variables and constraints.
Never invent, truncate, or replace required coefficients with dummy or random data.
Before optimization, validate coefficient dimensions and identifier coverage against decision index sets;
raise an explicit error for missing data instead of guessing, padding, or silently dropping entries.

[Your Gurobi Code]
"""

        code_gen_prompt = PromptTemplate(
            template=code_gen_template,
            input_variables=["query", "schema", "abstract_plan", "dataset_address", "few_shot_examples"]
        )
        
        code_gen_chain = LLMChain(llm=llm2, prompt=code_gen_prompt)
        
        final_code = code_gen_chain.run(
            query=user_query,
            schema=schema,
            abstract_plan=abstract_model_plan,
            dataset_address=dataset_address,
            few_shot_examples=few_shot_block_code 
        )
        
        if "```python" in final_code:
            code_block = final_code.split("```python", 1)[1]
            if "```" in code_block:
                code_block = code_block.split("```", 1)[0]
            final_answer = "```python\n" + code_block.strip() + "\n```"
        else:
            if not final_code.strip().startswith("import"):
                print(f"[Warning] Step 3 output was not a valid code block. Output: {final_code[:200]}...")
                final_answer = f"Error: Code generation failed. LLM returned non-code output:\n{final_code}"
            else:
                print("[Warning] Step 3 output missed ```python tag, adding it.")
                final_answer = "```python\n" + final_code.strip() + "\n```"
            
        print(f"[Final Answer]:\n{final_answer}")
        return {
            "formulation": abstract_model_plan, "code": final_answer, "observation": schema,
            "trace": {"status": "LEGACY_SCHEMA", "planner_attempt_count": 0,
                      "payload_hash": hashlib.sha256(schema.encode()).hexdigest()},
        }

    except Exception as e:
        print(f"[Gurobi Pipeline Error]: {e}")
        raise


def get_others_without_CSV_response(query):
    llm = make_llm()

    loader = CSVLoader(file_path=str(PROJECT_ROOT / "Large_Scale_Or_Files/RAG_Example_Others_Without_CSV.csv"), encoding="utf-8")
    documents = loader.load()

    embeddings = make_embeddings()
    vectors = FAISS.from_documents(documents, embeddings)

    retriever = vectors.as_retriever(search_kwargs={'k': 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    qa_tool = Tool(
        name="ORLM_QA",
        func=qa_chain.invoke,
        description=(
            "Use this tool to answer Querys."
            "Provide the Query as input, and the tool will retrieve the relevant information from the file and use it to answer the Query."
        ),
    )

    few_shot_examples = []
    similar_results = retrieve_similar_texts(query, retriever)

    for content in similar_results:

        split_at_formulation = content.split("Data_address:", 1)
        problem_description = split_at_formulation[0].replace("prompt:", "").strip()
        split_at_address = split_at_formulation[1].split("Label:", 1)
        split_at_label = split_at_address[1].split("Related:", 1)
        label = split_at_label[0].strip() 

        label = label.replace("{", "{{").replace("}", "}}")
        example = (
            "<EXAMPLE>\n"
            f"Question: {problem_description}\n\n"
            "Thought: Read the question and 1) identify the goal (minimize time/cost/crew or maximize throughput/value) and collect per-unit coefficients; 2) define decision variables and pick domains: counts of trips/units/vehicles are nonnegative integers, yes or no choices are binary, divisible flows or weights are nonnegative reals; 3) write the linear objective from the coefficients; 4) add constraints in this order: demand or target (exactly, at least, at most), capacity or supply upper bounds, flow conservation or stage linking across nodes or arcs, and any share or ratio limits rewritten as linear inequalities using the given per-unit rates, plus any minimum or maximum usage; 5) add nonnegativity and the chosen integrality or binary domains; 6) output only the LP: objective first, then constraints line by line with brief labels if needed, then a final line stating the variable domains."
            "Final Answer:\n"
            f"{label}\n"
            "</EXAMPLE>"
        )

        example = example.replace("{", "{{").replace("}", "}}")
        few_shot_examples.append(example)

    prefix = (
    f"{few_shot_examples}\n\n"
    """ 
    Use the following triggers to identify problem structures and apply the corresponding mathematical formulations.
    (1) INTEGER trigger:
    Question: A factory can run two machine types. How many of each machine should be installed given budget and space limits? Maximize output.
    Final Answer: 
    $\\max\\; p_1 x_1 + p_2 x_2$
    $\\text{{s.t. }} a_1 x_1 + a_2 x_2 \\le B,\\; s_1 x_1 + s_2 x_2 \\le S$
    $x_1, x_2 \\in \\mathbb{{Z}}_+.$
    (2) MULTI-PERIOD FLOW trigger:
    Question: Multi-period production with inventory and backorders. Costs for production/holding/backorder. Initial and terminal conditions given.
    Final Answer:
    \textbf{{Indices: }} t\\in T=\\{{1,\\dots,n\\}}. \\
    \textbf{{Given: }} d_t,\\ I_0,\\ B_0,\\ \\dots \\
    \textbf{{Vars: }} x_t\\ge0,\\ I_t\\ge0,\\ B_t\\ge0. \\
    \\min \\sum_t (c x_t + h I_t + p B_t) \\
    \text{{s.t. }} I_t - B_t = I_{{t-1}} - B_{{t-1}} + x_t - d_t,\\ \forall t \\
    I_n \\ge I^{{\\min}},\\ B_n=0.
    (3) LOGIC+BINARY trigger:
    Question: Choose exactly one option from set P and at least K items from set V, with quantities and budget.
    Final Answer:
    \textbf{{Sets: }} i\\in P,\\ j\\in V. \\ \textbf{{Vars: }} q_i,q_j\\ge0;\\ y_i\\in\\{{0,1\\}}, z_j\\in\\{{0,1\\}}.\\
    \\max \\sum_j f_j q_j \\
    \text{{s.t. }} \\sum_i y_i = 1,\\ \\sum_j z_j \\ge K \\
    0\\le q_i \\le M_i y_i,\\ \forall i;\\ \\ 0\\le q_j \\le M_j z_j,\\ \forall j \\
    \text{{Budget: }} \\sum_i c_i q_i + \\sum_j c_j q_j \\le B.
    """
    "USER QUESTION:\n{input}\n\n"
    "TASK:\n"
    "- Produce a complete LaTeX optimization model using ONLY information in the question.\n"
    "- Use a fixed structure INSIDE LaTeX: Indices/Sets; Given Parameters (convert all tables to arrays); Decision Variables (with domains); Objective; Constraints; Domain lines.\n"
    "- For multi-period problems, you MUST include state-balance recurrences and initial/terminal conditions explicitly.\n"
    "- Avoid nonlinear forms when possible: rewrite ratios/logic using linear constraints + binaries (big-M) with clearly defined M.\n\n"
    "You should decide VARIABLE–TYPE first!\n (If not mentioned or ambiguous, integer by default!)"
    "### FIRST RESPONSE FORMAT (exactly 3 lines) ###\n"
    "Thought: <brief>\n"
    "Action: ORLM_QA\n"
    "Action Input: {input}\n\n"
    "Begin."
)

    suffix = (
    "\n### AFTER OBSERVATION ###\n"
    "Respond with exactly two lines:\n"
    "Thought: <variable types: integer/binary/continuous>\n"
    "Final Answer: <ONLY LaTeX model. Must include: (i) indices/sets, (ii) parameter definitions (tables->arrays), (iii) explicit domain lines for EVERY variable, (iv) initial/terminal conditions if any. No prose.>\n"
    "Do NOT output anything else."
)

    agent = initialize_agent(
        tools=[qa_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={
            "prefix": prefix,
            "suffix": suffix,
            "input_variables": ["input"]
        },
        verbose=True,
        handle_parsing_errors=True,  # Enable error handling
    )

    openai.api_request_timeout = 60  
    query = query.replace('{','{{').replace('}','}}')
    output = agent.run({"input": query})

    return output
