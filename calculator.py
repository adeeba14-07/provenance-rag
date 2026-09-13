import re
from statistics import mean, median


def detect_calculation_intent(query):
    query_lower = query.lower()

    if any(w in query_lower for w in ["average", "mean", "avg"]):
        return "average"
    elif any(w in query_lower for w in ["sum", "total", "add up"]):
        return "sum"
    elif any(w in query_lower for w in ["maximum", "max", "highest", "largest", "biggest"]):
        return "max"
    elif any(w in query_lower for w in ["minimum", "min", "lowest", "smallest"]):
        return "min"
    elif any(w in query_lower for w in ["count", "how many", "number of"]):
        return "count"
    elif any(w in query_lower for w in ["median", "middle value"]):
        return "median"
    else:
        return None


def extract_relevant_column(query):
    """Try to identify which column the question is about (e.g., 'yes %', 'coverage')."""
    query_lower = query.lower()

    # Common column indicators
    column_keywords = {
        "yes": "yes",
        "no": "no",
        "coverage": "coverage",
        "rate": "rate",
        "percentage": "%",
        "percent": "%",
        "value": "value",
        "score": "score",
    }

    for keyword, column_hint in column_keywords.items():
        if keyword in query_lower:
            return column_hint

    return None


def extract_numbers_from_context(text, column_hint=None):
    """Extract numbers. If a column hint is given, filter by context."""
    if not text:
        return []

    # Split into lines
    lines = text.split("\n")
    numbers = []

    for line in lines:
        # Skip lines that look like headers
        if any(h in line.lower() for h in ["column", "header", "table", "chunk", "source:", "page:"]):
            continue

        # If we have a column hint, prefer lines containing it
        if column_hint and column_hint.lower() not in line.lower():
            # But still scan if the line has many numbers
            pass

        # Find numbers, but skip 0.0 and 1.0 (usually flags/booleans)
        found = re.findall(r'-?\d+\.?\d*', line)
        for f in found:
            try:
                num = float(f)
                # Skip years, skip flags 0.0 and 1.0 unless clearly a percentage
                if 1900 <= num <= 2099 and "." not in f:
                    continue
                if num in [0.0, 1.0]:
                    continue
                numbers.append(num)
            except ValueError:
                continue

    return numbers


def calculate(intent, numbers):
    if not numbers:
        return None, "No relevant numbers found to calculate."

    try:
        if intent == "average":
            return round(mean(numbers), 2), f"Average of {len(numbers)} values"
        elif intent == "sum":
            return round(sum(numbers), 2), f"Sum of {len(numbers)} values"
        elif intent == "max":
            return max(numbers), "Maximum value"
        elif intent == "min":
            return min(numbers), "Minimum value"
        elif intent == "count":
            return len(numbers), "Count of values"
        elif intent == "median":
            return round(median(numbers), 2), f"Median of {len(numbers)} values"
        else:
            return None, "Unknown calculation type."
    except Exception as e:
        return None, f"Calculation error: {e}"


def run_calculator(query, context_text):
    intent = detect_calculation_intent(query)
    if not intent:
        return None

    column_hint = extract_relevant_column(query)
    numbers = extract_numbers_from_context(context_text, column_hint)

    if not numbers:
        return {
            "intent": intent,
            "numbers_found": 0,
            "result": None,
            "explanation": "No relevant numeric values found. Note: this system does not perform calculations on table headers or flag values (0/1)."
        }

    result, explanation = calculate(intent, numbers)

    return {
        "intent": intent,
        "numbers_found": len(numbers),
        "result": result,
        "explanation": explanation,
        "numbers_used": numbers[:20]
    }