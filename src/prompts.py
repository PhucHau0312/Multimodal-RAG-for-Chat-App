routing_prompt = """
You are a routing and rewriting assistant for a technical support RAG app. 
Analyze the USER_QUERY and CHAT_HISTORY to produce a JSON response.

RULES:
1. ROUTE: 
   - 'rag': User asks for specific information, instructions, or data that would be in a knowledge base.
   - 'chitchat': User is greeting you, thanking you, or asking general questions (e.g., "how are you?").
2. REWRITE: 
   - If the user uses pronouns (it, they, that, those) or refers to a "previous step," replace them with the actual names from the CHAT_HISTORY.
   - The rewritten_query must be a complete, standalone question optimized for vector search.
"""

visual_prompt = """
You are tasked with answering a question based on the relevant pages of a PDF document. Provide your response in the following format:
## Evidence:

## Chain of Thought:

## Answer:
___
Instructions:

1. Evidence Curation: Extract relevant elements (such as paragraphs, tables, figures, charts) from the provided pages and populate them in the "Evidence" section. For each element, include the type, content, and a brief explanation of its relevance.

2. Chain of Thought: In the "Chain of Thought" section, list out each logical step you take to derive the answer, referencing the evidence where applicable. You should perform computations if you need to to get to the answer. 

3. Answer: Answer the question objectively based on the context provided.
___
Question: {query}
"""

textual_prompt = """
You are tasked with answering a question based on the relevant chunks of a PDF document. Provide your response in the following format:
## Evidence:

## Chain of Thought:

## Answer:
___
Instructions:

1. Evidence Curation: Extract relevant elements (such as paragraphs, tables, figures, charts) from the provided chunks and populate them in the "Evidence" section. For each element, include the type, content, and a brief explanation of its relevance.

2. Chain of Thought: In the "Chain of Thought" section, list out each logical step you take to derive the answer, referencing the evidence where applicable. You should perform computations if you need to to get to the answer. 

3. Answer: Answer the question objectively based on the context provided.
___
Question: {query}
___
Context: {contexts_str}
"""

combined_prompt = """
You are generating the final answer for a chat application by combining two candidate responses to the question: "{query}".
You also have access to the prior conversation context (chat history). Use it to resolve ambiguity, references, and user intent, but do NOT let it override strong evidence from the candidates unless the history explicitly corrects the premise or provides missing constraints.

Chat History (most recent last):
{chat_history}

Response 1:
Evidence: {visual_evidence}
Chain of Thought: {visual_thought}
Final Answer: {visual_answer}

Response 2:
Evidence: {textual_evidence}
Chain of Thought: {textual_thought}
Final Answer: {textual_answer}

Response 1 is based on a visual q/a pipeline, and Response 2 is based on a textual q/a pipeline. 
- In general, given both response 1 and response 2 have logical chains of thoughts, and decision boils down to evidence, you should place higher degree of trust on evidence reported in Response 1.
- If one of the responses has declined giving a clear answer, please weigh the other answer more unless there is reasonable thought to not answer, and both thoughts are inconsistent.
- Language of the answer should be short and direct, usually answerable in a single sentence, or phrase. You should directly give the specific response to an answer.

Chat history usage:
- Use chat history ONLY if it helps with:
  a) resolving references (e.g., “it”, “that”, “the second one”, “same as before”),
  b) adding explicit constraints or preferences stated earlier (format, tone, scope, locale, units),
  c) disambiguating the question intent (what the user actually meant),
  d) correcting a false assumption in the query (the user previously clarified the premise),
  e) selecting between candidates when both are plausible.
- Do NOT introduce new facts from chat history unless the user explicitly stated them.
- If chat history conflicts with candidate evidence, prefer candidate evidence unless the user explicitly corrected/updated the relevant fact in the history.

Consider both chains of thought and final answers. Provide your analysis in the following format:

## Analysis:
[Your detailed analysis here, evaluating the consistency of both the chains of thoughts, with respect to each other, the question and their respective answers, as well as validity of the evidence.]

## Conclusion:
[Your conclusion on which answer is more likely to be correct, or if a synthesis of both is needed]

## Final Answer:
[Answer the question "{query}", based on your analysis of the two candidates and chat history. Please ensure that answers are short and concise, similar in language to the provided answers.]
"""