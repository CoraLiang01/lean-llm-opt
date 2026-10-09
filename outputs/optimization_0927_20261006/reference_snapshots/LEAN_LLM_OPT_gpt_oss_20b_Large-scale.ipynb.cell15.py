class OllamaFormulationReActOutputParser(AgentOutputParser):
    """Require CSVQA once while accepting gpt-oss's direct final model."""

    fallback_input: str
    tool_seen: bool = False

    def parse(self, text):
        try:
            parsed = ReActSingleInputOutputParser().parse(text)
        except Exception as original_error:
            output = str(text or "").strip()
            if not self.tool_seen:
                self.tool_seen = True
                return AgentAction(tool="CSVQA", tool_input=self.fallback_input, log=text)
            if not output:
                raise OutputParserException(
                    "CSVQA has already been called. Return the complete formulation as Final Answer.",
                    observation="Use the existing CSVQA Observation and return Final Answer now.",
                    llm_output=text,
                    send_to_llm=True,
                )
            return AgentFinish(return_values={"output": output}, log=text)
        if isinstance(parsed, AgentAction):
            if self.tool_seen:
                raise OutputParserException(
                    "CSVQA has already been called exactly once. Do not call any tool again; "
                    "write the complete mathematical formulation as the Final Answer now.",
                    observation=(
                        "CSVQA has already returned the required data. Use the existing Observation "
                        "and return the complete Final Answer without another Action."
                    ),
                    llm_output=text,
                    send_to_llm=True,
                )
            self.tool_seen = True
            return AgentAction(tool="CSVQA", tool_input=self.fallback_input, log=text)
        if isinstance(parsed, AgentFinish) and not self.tool_seen:
            self.tool_seen = True
            return AgentAction(tool="CSVQA", tool_input=self.fallback_input, log=text)
        return parsed

    def get_format_instructions(self):
        return ReActSingleInputOutputParser().get_format_instructions()


def formulate_with_csvqa(query, dataset_address, route, system_prompt, tool_description, prefix, suffix):
    """Run the shared CSVQA agent with the route's prompt and fixed data mode."""
    llm, qa_tool, csvqa_result = build_csvqa_components(
        dataset_address, system_prompt, tool_description, route=route, user_query=query,
    )
    agent = initialize_agent(
        tools=[qa_tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={"prefix": prefix, "suffix": suffix, "output_parser": OllamaFormulationReActOutputParser(fallback_input=str(query))}, verbose=True,
        handle_parsing_errors=True, max_iterations=4, return_intermediate_steps=True,
    )
    result = invoke_react_with_required_csvqa(agent, query, route)
    formulation = str(result.get("output", "") or "").strip()
    incomplete = not formulation or "Agent stopped due to iteration limit or time limit" in formulation
    if incomplete:
        observation = str(csvqa_result.get("observation", "") or "").strip()
        if not observation:
            raise RuntimeError("CSVQA completed without an observation")
        completion_prompt = f"""You are completing the Final Answer of the same {route} ReAct
formulation task. CSVQA has already been called exactly once. Do not call tools and do not discuss
the protocol. Use the original query as the authority and the exact CSVQA Observation below as the
only data source. Return only a complete mathematical optimization formulation, including index
sets, parameters, variables and domains, the exact objective, every query constraint, and an exact
mapping to the returned table IDs and columns. Never copy values from examples or invent data.

Original query:
{query}

CSVQA Observation:
{observation}
"""
        formulation = str(llm.invoke([HumanMessage(content=completion_prompt)]).content or "").strip()
        if not formulation:
            raise RuntimeError("Formulation finalization returned an empty answer")
    return {"formulation": formulation, **csvqa_result}


@lru_cache(maxsize=None)
def get_route_retriever(route, k):
    """Filter the shared RAG table by workflow route."""
    if route not in {"NRM", "RA", "TP", "AP", "FLP"}:
        raise ValueError(f"Unknown workflow route: {route}")
    documents = []
    for row in _unified_rag_rows():
        row_route = "FLP" if row["Type"] == "UFLP" else row["Type"]
        if row_route != route:
            continue
        content = "\n".join(f"{field}: {row[field]}" for field in
                            ("prompt", "Data_address", "Label", "Related"))
        documents.append(Document(page_content=content))
    if not documents:
        raise ValueError(f"No unified RAG examples for route {route}")
    return FAISS.from_documents(documents, make_embeddings()).as_retriever(
        search_kwargs={"k": min(k, len(documents))}
    )


def retrieve_similar_texts(query, retriever):
    return [doc.page_content for doc in retriever.invoke(query)]


def get_NRM_response(query, dataset_address):
    retrieve='product'
    retriever = get_route_retriever("NRM", 1)
    few_shot_examples = []

    similar_results = retrieve_similar_texts(query, retriever)

    for content in similar_results:
        split_at_formulation = content.split("Data_address:", 1)
        problem_description = split_at_formulation[0].replace("prompt:", "").strip()  
        split_at_address = split_at_formulation[1].split("Label:", 1)
        data_address = split_at_address[0].strip()

        split_at_label = split_at_address[1].split("Related:", 1)
        label = split_at_label[0].strip()  
        Related = split_at_label[1].strip()
        information = read_csv_compat(data_address)
        information_head = information[:36]

        example_data_description = "\nHere is the product data:\n"
        for i, r in information_head.iterrows():
            example_data_description += f"Product {i + 1}: {r['Product Name']}, revenue w_{i + 1} = {r['Revenue']}, demand rate a_{i + 1} = {r['Demand']}, initial inventory c_{i + 1} = {r['Initial Inventory']}\n"


        label = label.replace("{", "{{").replace("}", "}}")
        few_shot_examples.append(fr"""

Question: Based on the following problem description and data, please formulate a complete mathematical model using real data from retrieval. {problem_description}

Thought: I need to formulate the objective function and constraints of the linear programming model based on the user's description and the provided data. I should retrieve the relevant information from the CSV file. Pay attention: 1. If the data to be retrieved is not specified, retrieve the whole dataset instead. 2. I should pay attention if there is further detailed constraint in the problem description. If so, I should generate additional constraint formula. 3. The final expressions should not be simplified or abbreviated.

Action: CSVQA

Action Input: Retrieve all the {retrieve} data {Related} to formulate the mathematical model with no simplification or abbreviation. Retrieve the documents in order, row by row. Use the given context to answer the question. If mention a certain kind of product, retrieve all the relavant product information detail judging by its product name. If not mention a certain kind of product, retrieve all the data instead. Only present final answer in details of row, instead of giving a sheet format.

Observation: {example_data_description}

Thought: Now that I have the necessary data, construct the objective function and constraints using the retrieved data as parameters of the formula. Ensure to include any additional detailed constraints present in the problem description. Always pay attention to the variable type. If not mentioned, use nonnegative integer. Do NOT include any explanations, notes, or extra text. Format the expressions strictly in markdown ONLY in this exact format: {label}. Following this example. The expressions should not be simplified or abbreviated. Besides, I need to use the $$ or $ to wrap the mathematical expressions instead of \[, \], \( or \). I also should avoid using align, align* and other latex environments. Besides, I should also avoid using \begin, \end, \text.

Final Answer: 
{label}
""")

    few_shot_examples = "\n\n".join(few_shot_examples)
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
