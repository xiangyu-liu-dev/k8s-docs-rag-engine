from rag.local_llm import LocalLLM


# ==================== Retrieval Metrics ====================


def is_relevant(result, answer_refs):
    """Check if result contains any reference answer"""
    text = (
        ((result.get("heading") or "") + " " + (result.get("text") or ""))
        .replace("\n", "")
        .lower()
    )

    for ref in answer_refs:
        if ref.lower() in text:
            return True
    return False


def hit_at_k(results, answer_refs, k=5):
    """Did we find at least one relevant doc in top-k?"""
    for r in results[:k]:
        if is_relevant(r, answer_refs):
            return 1
    return 0


def mrr(results, answer_refs, k=5):
    """Mean Reciprocal Rank - where is the first relevant doc?"""
    for i, r in enumerate(results[:k]):
        if is_relevant(r, answer_refs):
            return 1 / (i + 1)
    return 0.0


# ==================== Generation Metrics (LLM-based) ====================


class RAGEvaluator:
    def __init__(self, llm: LocalLLM):
        self.llm = llm

    def judge_faithfulness(self, query: str, answer: str, context: str) -> float:
        """Judge if answer is fully supported by context"""

        user_message = f"""Evaluate if the answer is supported by the context.

Context:
{context}

Question:
{query}

Answer:
{answer}

Criteria:
- The core claims should be supported by the context
- Reasonable paraphrasing and summarization is acceptable
- If answer says "I don't know", that's YES if context lacks info

Is the answer supported by the context?"""

        verdict = self.llm.generate_judgment(user_message)
        return 1.0 if verdict == "YES" else 0.0

    def judge_answer_relevance(self, query: str, answer: str) -> float:
        """Judge if answer addresses the question"""

        user_message = f"""Does the answer directly address the question?

Question: {query}

Answer: {answer}

Criteria:
- Answer should respond to what was asked
- "I don't know" is YES if appropriate

Does the answer address the question?"""

        verdict = self.llm.generate_judgment(user_message)
        return 1.0 if verdict == "YES" else 0.0

    def judge_answer_quality_with_refs(
        self, query: str, answer: str, answer_refs: list[str]
    ) -> float:
        """Judge if answer contains key information from reference snippets"""
        refs_text = "\n".join(f"- {ref}" for ref in answer_refs)

        user_message = f"""Does the answer cover the key information from the reference points?

Question: {query}

Reference points:
{refs_text}

Answer: {answer}

Criteria:
- Answer should convey main points from references
- Exact wording not required
- Answer of "How" doesn't need to contain every point

Does the answer cover the key points?"""

        verdict = self.llm.generate_judgment(user_message)
        return 1.0 if verdict == "YES" else 0.0

    def evaluate_single(
        self, query: str, answer: str, context: str, answer_refs: list[str] = None
    ) -> dict[str, float]:
        """Evaluate a single RAG response"""

        results = {
            "faithfulness": self.judge_faithfulness(query, answer, context),
            "answer_relevance": self.judge_answer_relevance(query, answer),
        }

        if answer_refs:
            results["answer_quality"] = self.judge_answer_quality_with_refs(
                query, answer, answer_refs
            )

        results["overall"] = sum(results.values()) / len(results)

        return results


# ==================== Answer Generation ====================


def generate_answer(query: str, context: str, llm: LocalLLM) -> str:
    """Generate answer using RAG with system prompt"""
    user_message = f"""Context:
{context}

Question: {query}

Instructions:
- Use ONLY information from the context
- If context lacks info, say: "I don't know"

Answer:"""

    return llm.generate_answer(user_message)
