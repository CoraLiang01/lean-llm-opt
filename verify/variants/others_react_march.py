import re
import pandas as pd
import numpy as np
import os
from typing import Union
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate

def escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")

OTHERS_RAG_PATH = "Large_Scale_Or_Files/RAG_Example_Others.csv"
embeddings = EMBEDDINGS
Others_docs = CSVLoader(file_path=OTHERS_RAG_PATH, encoding="utf-8").load()
Others_store: FAISS = FAISS.from_documents(Others_docs, embeddings)


def retrieve_examples(store, query: str, k: int = None):
    # k comes from exp_config.yaml:retriever_k.examples_others (8, the March
    # value) instead of the literal that used to sit here, so the number is
    # recorded in config.json with everything else. An explicit k still wins.
    retriever = (store.as_retriever(search_kwargs={"k": k}) if k is not None
                 else lx.build_retriever(CFG, store, "examples_others"))
    return retriever.invoke(query)


def build_few_shot_Other(store, user_query: str, k: int = None, retrieve: str = "all data"):
    """
    Few-shot block for the Others-with-CSV route, as the March pipeline built it.

    This replaces the "Abstract Model Plan" builder that stood here from
    2026-08-03. That version was ported over from the gpt-oss-20b notebook and
    differed from the pipeline the 85.1% was measured on in three ways that all
    push the same direction:

      * k=3 exemplars instead of 8;
      * each exemplar showed a placeholder schema (`col1, col2, col3 / 1, 2, 3`)
        instead of the reference instance's actual numbers;
      * the exemplar's Final Answer was Gurobi code, so the route emitted code
        directly and never produced a mathematical model.

    Reference-file note: the March RAG file had columns
    (prompt, Data_address, Label, Related) and the code split the tail off at
    "Related:". RAG_Example_Others.csv now carries
    (prompt, Data_address, Label, Label_Code, Answer) instead, so the split has
    to land on Label_Code: to recover the same `label` -- the mathematical
    model alone. `Related` was blank in every March row (the saved notebook
    output prints it as ' '), so dropping it changes nothing.
    """
    few_shot_examples = []
    similar_results = retrieve_similar_docs(
        user_query,
        (store.as_retriever(search_kwargs={"k": k}) if k is not None
         else lx.build_retriever(CFG, store, "examples_others")))

    for i, result in enumerate(similar_results, 1):
        content = result['content']
        try:
            split_at_formulation = content.split("Data_address:", 1)
            problem_description = split_at_formulation[0].replace("Query:", "").strip()
            if problem_description == user_query:
                continue                      # leave-one-out, as in March

            split_at_address = split_at_formulation[1].split("Label:", 1)
            data_address = split_at_address[0].strip()
            # March: .split("Related:", 1)[0]  -- see the docstring
            label = re.split(r'\n(?:Label_Code|Related|Answer):',
                             split_at_address[1])[0].strip()
            Related = ''                      # blank in every March row

            datas = data_address.split()
            information = [pd.read_csv(d) for d in datas]
            example_data_description = "\nHere is the data:\n"
            for df_index, df in enumerate(information):
                for z, r in df.iterrows():
                    example_data_description += " ".join(
                        [f"{col} = {r[col]}" for col in df.columns]) + "\n"
        except Exception as e:
            print(f"[build_few_shot_Other Error] Failed to parse example: {e}. "
                  f"Content: {content[:50]}")
            continue

        label = label.replace("{", "{{").replace("}", "}}")

        example = f"""
Question: Based on the following problem description and data, please formulate a complete mathematical model using real data from retrieval. {problem_description}

Thought: I need to formulate the objective function and constraints of the linear programming model based on the user's description and the provided data. I should retrieve the relevant information from the CSV file. If the data to be retrieved is not specified, retrieve the whole dataset instead. I should pay attention if there is further detailed constraint in the problem description. If so, I should generate additional constraint formula. The final expressions should not be simplified or abbreviated.

Action: CSVQA

Action Input: Retrieve all the {retrieve} data related to {Related} to formulate the mathematical model with no simplification or abbreviation.

Observation: {example_data_description}

Thought: Now that I have the necessary data, I would construct the objective function and constraints using the retrieved data as parameters of the formula. I should pay attention if there is further detailed constraint in the problem description. If so, I should generate additional constraint formula according to the retrieved 'product id'.  Respond ONLY in this exact format: {label}. Do NOT include any explanations, notes, or extra text. The expressions should not be simplified or abbreviated.

Final Answer:
{label}
"""
        # Escaped a second time, exactly as the March cell did: `label` is
        # already brace-escaped above, so it reaches the prompt template as
        # {{...}} and renders as literal {...} after one format pass. Kept
        # verbatim rather than "fixed" -- this is the prompt the 85.1% was
        # produced with, and changing it here would mean the port no longer
        # tests the thing it is meant to test.
        example = example.replace("{", "{{").replace("}", "}}")
        few_shot_examples.append(example)

    return few_shot_examples


def get_Others_response(query, dataset_address):
    """Others-with-CSV route: ReAct agent over a CSVQA tool, as of March."""
    retrieve = "all data"
    few_shot_examples = build_few_shot_Other(Others_store, query, retrieve=retrieve)

    dfs = []
    file_addresses = dataset_address.strip().split('\n')
    df_index = 0
    data_description = " "
    for file_address in file_addresses:
        try:
            df = pd.read_csv(file_address)
            file_name = file_address.split('/')[-1]
            if 'demand' in df.columns:
                result = df['demand'].values.tolist()
                data_description += "d=" + str(result) + "\n"
            elif 'fixed_costs' in df.columns:
                result = df['fixed_costs'].values.tolist()
                data_description += "c=" + str(result) + "\n"
            else:
                for row_idx, row in df.iterrows():
                    data_description += ", ".join(
                        [f"{col} = {row[col]}" for col in df.columns])
            df_index += 1
            dfs.append((file_name, df))
        except Exception as e:
            print(f"Error reading file {file_address}: {e}")

    embeddings = EMBEDDINGS
    vectors = FAISS.from_texts([data_description], embeddings)
    retriever = lx.build_retriever(CFG, vectors, "data_others")
    llm2 = lx.build_llm(CFG, "modeler")

    system_prompt = (
        "Retrieve the documents in order from top to bottom. Use the retrieved context to answer the question. If mention a certain kind of product, retrieve all the relavant product information detail judging by its product name. If not mention a certain kind of product, retrieve all the data instead."
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm2, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    def qa_wrapper(query: str):
        return qa_chain.invoke({"input": query})['answer']

    qa_tool = Tool(
        name="CSVQA",
        func=qa_wrapper,
        description="Use this tool to answer Querys based on the provided CSV data and retrieve product data similar to the input query."
    )

    prefix = f"""You are an assistant that generates a mathematical models based on the user's description and provided CSV data.

    Please refer to the following example and generate the answer in the same format:

    {few_shot_examples}

    When you need to retrieve information from the CSV file, use the provided tool.

    """

    suffix = """

    Begin!

    User Description: {input}
    {agent_scratchpad}"""

    agent2 = initialize_agent(
        tools=[qa_tool],
        llm=llm2,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        **lx.agent_kwargs(CFG, prefix, suffix),
    )
    query = query.replace("{", "{{").replace("}", "}}")

    result = agent2.invoke(query)
    output = result['output']
    return output
