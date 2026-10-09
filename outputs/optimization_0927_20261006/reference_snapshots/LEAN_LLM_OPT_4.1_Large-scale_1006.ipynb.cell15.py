def formulate_with_csvqa(query, dataset_address, route, system_prompt, tool_description, prefix, suffix):
    """Original ReAct with mandatory current CSVQA and user-authorized protocol restarts."""
    route = normalize_route(route)
    if CSVQA_MODE_BY_ROUTE[route] == "planned":
        system_prompt = CSVQA_PLANNED_PROMPTS[route]
        tool_description = CSVQA_TOOL_DESCRIPTIONS[route]
        prefix = prefix.replace("please output required parameters in a whole text, including all vectors and matrices.",
                                "return symbolic parameters and an exact source Data Mapping; do not enumerate values.")
    llm, qa_tool, csvqa_result = build_csvqa_components(
        dataset_address, system_prompt, tool_description, route=route, user_query=query,
    )
    tool_history, current_observations = [], []
    original_tool_func = qa_tool.func
    def invoke_csvqa(tool_query):
        entry = {"request": str(tool_query)}
        tool_history.append(entry)
        observation = original_tool_func(tool_query)
        entry.update(trace=dict(csvqa_result.get("trace", {})), observation=observation)
        if CSVQA_MODE_BY_ROUTE[route] == "planned":
            current_observations.append(json.loads(observation))
        return observation
    qa_tool.func = invoke_csvqa
    sources = dict(enumerate(raw.strip() for raw in str(dataset_address).splitlines() if raw.strip()))
    prefix += ("\nFor the CURRENT problem, call CSVQA at least once before the Final Answer. "
               "Demonstration Observations are historical and cannot supply current data. "
               "You may call CSVQA multiple times for different current CSV files. "
               "Use the original user description as Action Input to read all CSV files, or use "
               'JSON Action Input with keys "query" and "file_indices" to read selected zero-based indices. '
               "Use a concise symbolic model and exact Data Mapping with table_id and column names. "
               "Bind every parameter to source data or a query-defined expression; do not invent a missing limit. "
               "If total capacity denotes listed per-entity capacities, retain their bounds and their sum. "
               "Preserve all variable domains, constraints, objective sense and additive constants. "
               "Define each index set from ALL current returned entities unless the original query restricts it. "
               "Example counts and profile samples must not determine the current entity set. "
               "Use optional matrix alias mappings only when supplied; otherwise preserve exact raw IDs.\n"
               "Current source index-to-path mapping: " + escape_braces(json.dumps(sources)))
    prefix += '\nApply unconditional quantity/resource bounds unconditionally. Only use activation-conditioned bounds when the current query explicitly makes them conditional; an activation fee alone does not make an otherwise unconditional bound conditional.'
    suffix = """Begin!

CURRENT USER DESCRIPTION START
{input}
CURRENT USER DESCRIPTION END

CURRENT CSVQA DATA STATE: {current_data_state}

If the state is NOT_LOADED, a Final Answer is forbidden. Respond with:
Thought: I need the current source data.
Action: CSVQA
Action Input: <copy only the user description between START and END, or provide the specified JSON>
Do not copy markers, protocol instructions, examples or status into Action Input.
When the state is READY, request another CSV if needed, or respond with this envelope:
Thought: I have the required current data and can formulate the model.
Final Answer:
<the complete symbolic mathematical model and Data Mapping>
The literal label "Final Answer:" is mandatory before every mathematical-model heading.
Never output a bare model or markdown heading before these labels.
Keep numerical data in the Observation; do not repeat its tables or enumerate parameters.
Do not treat any demonstration's Observation as current problem data.
{agent_scratchpad}"""
    def create_agent():
        csvqa_result.clear()
        current_observations.clear()
        retry_note = ("\nA previous protocol attempt was discarded. Follow the exact Thought/Action/Action Input "
                      "or Thought/Final Answer envelope and call CSVQA before your final model."
                      if REACT_PROTOCOL_EVENTS else "")
        agent = initialize_agent(
            tools=[qa_tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_kwargs={"prefix": prefix + retry_note, "suffix": suffix}, verbose=True,
            handle_parsing_errors=False, return_intermediate_steps=True, early_stopping_method="force",
        )
        agent.agent.llm_chain.prompt = agent.agent.llm_chain.prompt.partial(
            current_data_state=lambda: "READY" if csvqa_result.get("observation") else "NOT_LOADED",
        )
        return agent
    try:
        result = invoke_react_with_required_csvqa(create_agent, query, route)
        if len(current_observations) > 1:
            tables = {t["table_id"]: t for payload in current_observations for t in payload["tables"]}
            relationships = {r["matrix_table_id"]: r for payload in current_observations
                             for r in payload.get("relationships", [])}
            payload = {**current_observations[-1], "tables": list(tables.values()),
                       "relationships": list(relationships.values()), "ignored_file_indices": []}
            csvqa_result["observation"] = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        evidence = dict(csvqa_result)
        evidence["trace"] = {**evidence.get("trace", {}), "formulation_protocol": "ReAct",
                             "csvqa_call_count": len(tool_history), "tool_calls": tool_history,
                             **react_protocol_record_fields()}
        exc.csvqa_result = evidence
        if not evidence.get("observation"):
            exc.pipeline_stage = "data_extraction"
        raise
    trace = csvqa_result.setdefault("trace", {})
    trace.update(formulation_protocol="ReAct", csvqa_call_count=len(tool_history), tool_calls=tool_history,
                 fallback_count=sum(c.get("trace", {}).get("fallback_count", 0) for c in tool_history),
                 **react_protocol_record_fields())
    trace["payload_hash"] = hashlib.sha256(csvqa_result["observation"].encode("utf-8")).hexdigest()
    return {"formulation": result["output"], **csvqa_result}


@lru_cache(maxsize=1)
def _load_rag_table():
    """Read the unified examples once, preserving every field as text."""
    frame = read_csv_compat(RAG_EXAMPLES_ALL_PATH, dtype=str, keep_default_na=False)
    required = {"prompt", "Data_address", "Related", "Required Data", "Label", "Code", "Type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"RAG example CSV is missing columns: {sorted(missing)}")
    frame["Type"] = frame["Type"].map(normalize_route)
    return frame


def load_rag_examples(route):
    """Return one canonical route; UFLP maps to FLP and Mixture to Others."""
    frame = _load_rag_table()
    examples = frame.loc[frame["Type"].eq(normalize_route(route))].copy()
    if examples.empty:
        raise ValueError(f"No RAG examples found for route {route!r}")
    return examples


@lru_cache(maxsize=None)
def _get_rag_store(route):
    """Share a semantic index across modeling and code generation for each route."""
    documents = [
        Document(
            page_content=example["prompt"],
            metadata={"example_json": json.dumps(example, ensure_ascii=False)},
        )
        for example in load_rag_examples(route).to_dict("records")
    ]
    # Embed the problem only: code/formulation length must not dominate similarity.
    return FAISS.from_documents(documents, make_embeddings())


def retrieve_rag_examples(route, query, k=1):
    """Retrieve structured example dictionaries, never parse CSVLoader display text."""
    if k < 1:
        return []
    route = normalize_route(route)
    documents = _get_rag_store(route).similarity_search(
        str(query), k=min(k, len(load_rag_examples(route))),
    )
    return [json.loads(document.metadata["example_json"]) for document in documents]


def retrieve_similar_texts(query, retriever):
    # The separate query-only reference dataset still uses its original retriever.
    return [doc.page_content for doc in retriever.invoke(query)]


def escape_braces(text):
    """Protect literal example braces when embedding text into an agent template."""
    return text.replace("{", "{{").replace("}", "}}")


def rag_example_observation(example):
    """Prefer the curated evidence, including its original source-row identifiers."""
    evidence = example["Required Data"].strip()
    if evidence:
        return evidence
    blocks = []
    for raw in example["Data_address"].splitlines():
        if raw.strip():
            frame = read_csv_compat(raw.strip(), dtype=str, keep_default_na=False)
            blocks.append(f"Source: {raw.strip()}\n{frame.to_json(orient='records', force_ascii=False)}")
    return "\n\n".join(blocks)


def build_formulation_examples(route, query, k=1):
    """Render the common ReAct example format once for all CSV modeling routes."""
    examples = []
    for example in retrieve_rag_examples(route, query, k):
        block = f"""Question: Based on the following problem description and data, formulate a complete mathematical model. {example['prompt']}

Thought: Retrieve all data needed by the question, in source order. Preserve identifiers, coefficients, and any additional constraints. If the question does not request a subset, retrieve the whole dataset.

Action: CSVQA

Action Input: Retrieve all data {example['Related'].strip()} required to formulate this model. Preserve source order and exact identifiers and values, without simplification or abbreviation.

Observation: {rag_example_observation(example)}

Thought: Use the retrieved data as parameters and include all query-specific constraints. Choose variable domains from the question. Return the complete mathematical model in markdown, with $ or $$ for mathematical expressions and without extra explanations or LaTeX environments.

Final Answer:
{example['Label']}"""
        # Prefix is itself a PromptTemplate; escape the entire block exactly once.
        examples.append(escape_braces(block))
    return "\n\n".join(examples)


def get_NRM_response(query, dataset_address):
    few_shot_examples = build_formulation_examples("NRM", query, k=1)
    csvqa_system_prompt = CSVQA_PLANNED_PROMPTS["NRM"]
    csvqa_tool_description = CSVQA_TOOL_DESCRIPTIONS["NRM"]

    route_instruction = """Call CSVQA exactly once and return an ABSTRACT model. Define
index sets, parameters, variables, objective, and constraints symbolically. Include a short Data
Mapping that identifies every source by exact table_id and exact column name. Do not copy record
values or literal record counts."""
    prefix = f"""You are an assistant that formulates mathematical optimization models.

    The examples below illustrate model structure only; never copy their data.

    {few_shot_examples}

    {route_instruction}
    Use the user query as the authority for objective sense, variable domains, boundary semantics,
    and additional constraints. Do not invent missing data or silently omit returned data.
    """

    suffix = """

    Begin!

    User Description: {input}
    {agent_scratchpad}"""

    return formulate_with_csvqa(
        query, dataset_address, "NRM", csvqa_system_prompt, csvqa_tool_description, prefix, suffix,
    )
