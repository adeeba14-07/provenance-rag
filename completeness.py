def check_term_in_full_document(term, collection):
    """Search the ENTIRE collection for a term, not just top-K chunks."""
    if not term or len(term) < 3:
        return False, 0

    try:
        all_data = collection.get(include=["documents"])
        all_docs = all_data.get("documents", [])

        count = 0
        term_lower = term.lower()

        for doc in all_docs:
            if term_lower in doc.lower():
                count += 1

        return count > 0, count
    except Exception:
        return False, 0


def extract_key_terms_from_question(question):
    """Extract proper nouns and key terms from the question."""
    import re

    # Words starting with capital letters (proper nouns)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', question)

    # Remove common words
    stopwords = {"What", "Where", "When", "Who", "Which", "How", "Why", "Is", "Are", "The", "Does", "Do", "Can"}

    terms = [w for w in proper_nouns if w not in stopwords]
    return list(set(terms))[:5]


def verify_absence_claims(answer, question, collection):
    """Before accepting an answer that claims something is missing, check the full document."""
    answer_lower = answer.lower()

    # Check if the answer makes an absence claim
    absence_phrases = [
        "not found", "not present", "not in the document", "not mentioned",
        "does not contain", "no information", "no data", "absent",
        "cannot find", "not available", "is missing"
    ]

    makes_absence_claim = any(phrase in answer_lower for phrase in absence_phrases)

    if not makes_absence_claim:
        return True, "No absence claim made."

    # Extract what the system claims is absent
    terms = extract_key_terms_from_question(question)

    for term in terms:
        found, count = check_term_in_full_document(term, collection)
        if found:
            return False, f"'{term}' was not in the retrieved chunks, but it DOES appear {count} times in the full document."

    return True, "Absence claim verified against full document."