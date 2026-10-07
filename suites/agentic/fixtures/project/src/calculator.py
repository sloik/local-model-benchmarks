"""Simple arithmetic calculator with left-to-right evaluation."""

def calculate(expression: str) -> float:
    """Evaluate a simple arithmetic expression with left-to-right evaluation.

    Supports +, -, *, / operators with float and integer operands.
    Does not respect operator precedence; evaluates strictly left-to-right.

    Args:
        expression: A string containing the arithmetic expression.

    Returns:
        The result of the evaluation as a float.

    Raises:
        ValueError: If the expression is invalid or contains division by zero.

    Examples:
        >>> calculate("2 + 3")
        5.0
        >>> calculate("10 - 4")
        6.0
        >>> calculate("10 + 5 * 2")  # Left-to-right: (10 + 5) * 2 = 30
        30.0
    """
    if expression is None:
        raise ValueError("Expression not set")

    expression = expression.strip()

    if not expression:
        raise ValueError("Empty expression")

    # Tokenize inline: split into operands and single-character operators.
    tokens = []
    current_token = ""
    operators = {"+", "-", "*", "/"}

    for char in expression:
        if char in operators:
            if current_token.strip():
                tokens.append(current_token.strip())
            tokens.append(char)
            current_token = ""
        elif char.isspace():
            if current_token.strip():
                tokens.append(current_token.strip())
            current_token = ""
        else:
            current_token += char

    if current_token.strip():
        tokens.append(current_token.strip())

    if not tokens:
        raise ValueError("Empty expression")

    # Evaluate inline, validating operand/operator positions left-to-right.
    result = None
    pending_op = None

    for i, token in enumerate(tokens):
        if i % 2 == 0:  # operand position
            try:
                operand = float(token)
            except ValueError:
                raise ValueError(f"Invalid operand: {token}")
            if result is None:
                result = operand
            else:
                if pending_op == "+":
                    result = result + operand
                elif pending_op == "-":
                    result = result - operand
                elif pending_op == "*":
                    result = result * operand
                elif pending_op == "/":
                    if operand == 0:
                        raise ValueError("Division by zero")
                    result = result / operand
                pending_op = None
        else:  # operator position
            if token not in operators:
                raise ValueError(f"Invalid operator: {token}")
            pending_op = token

    # A trailing operator with no operand is an invalid structure.
    if pending_op is not None:
        raise ValueError("Invalid expression structure")

    return result
