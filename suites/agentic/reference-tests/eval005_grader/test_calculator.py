"""Tests for the calculator module."""

import pytest

from calculator import calculate, Calculator

class TestBasicOperations:
    """Test basic arithmetic operations."""

    def test_addition(self):
        """Test simple addition."""
        assert calculate("2 + 3") == 5.0

    def test_subtraction(self):
        """Test simple subtraction."""
        assert calculate("10 - 4") == 6.0

    def test_multiplication(self):
        """Test simple multiplication."""
        assert calculate("3 * 7") == 21.0

    def test_division(self):
        """Test simple division."""
        result = calculate("10 / 3")
        assert abs(result - 3.333333333333) < 0.0001

class TestChainedOperations:
    """Test expressions with multiple operations."""

    def test_chained_addition(self):
        """Test multiple additions."""
        assert calculate("1 + 2 + 3") == 6.0

    def test_left_to_right_no_precedence(self):
        """Test that evaluation is left-to-right, not by operator precedence."""
        # Normal precedence would be (10 + 5) * 2 = 30, but left-to-right is correct
        assert calculate("10 + 5 * 2") == 30.0

class TestFloatOperands:
    """Test operations with floating-point numbers."""

    def test_float_addition(self):
        """Test addition with floats."""
        assert calculate("1.5 + 2.5") == 4.0

    def test_float_division(self):
        """Test division resulting in float."""
        result = calculate("5.0 / 2.0")
        assert abs(result - 2.5) < 0.0001

class TestSingleNumber:
    """Test with a single operand."""

    def test_single_integer(self):
        """Test single number."""
        assert calculate("42") == 42.0

    def test_single_float(self):
        """Test single float number."""
        assert calculate("3.14") == 3.14

class TestErrorHandling:
    """Test error cases."""

    def test_division_by_zero(self):
        """Test that division by zero raises ValueError."""
        with pytest.raises(ValueError, match="Division by zero"):
            calculate("5 / 0")

    def test_invalid_operand(self):
        """Test that invalid operands raise ValueError."""
        with pytest.raises(ValueError, match="Invalid operand"):
            calculate("2 + abc")

    def test_invalid_operator(self):
        """Test that invalid operators raise ValueError."""
        with pytest.raises(ValueError, match="Invalid operator"):
            calculate("2 % 3")

    def test_empty_expression(self):
        """Test that empty expression raises ValueError."""
        with pytest.raises(ValueError, match="Empty expression"):
            calculate("")

    def test_only_spaces(self):
        """Test that whitespace-only expression raises ValueError."""
        with pytest.raises(ValueError, match="Empty expression"):
            calculate("   ")

    def test_invalid_structure(self):
        """Test that invalid token structure raises ValueError."""
        with pytest.raises(ValueError, match="Invalid operand"):
            calculate("2 + + 3")

class TestCalculatorClass:
    """Test the Calculator class methods."""

    def test_parse(self):
        """Test that parse() method works correctly."""
        calc = Calculator()
        tokens = calc.parse("2 + 3")
        assert isinstance(tokens, list)
        assert tokens == ["2", "+", "3"]

    def test_evaluate(self):
        """Test that evaluate() method works correctly."""
        calc = Calculator()
        tokens = ["2", "+", "3"]
        result = calc.evaluate(tokens)
        assert result == 5.0

    def test_parse_and_evaluate(self):
        """Test that parse and evaluate work together."""
        calc = Calculator()
        tokens = calc.parse("2 + 3")
        result = calc.evaluate(tokens)
        assert result == 5.0

    def test_parse_returns_token_list(self):
        """Test that parse returns a list of tokens."""
        calc = Calculator()
        tokens = calc.parse("2 + 3")
        assert isinstance(tokens, list)
        assert tokens == ["2", "+", "3"]

    def test_parse_with_floats(self):
        """Test that parse handles float operands."""
        calc = Calculator()
        tokens = calc.parse("1.5 + 2.5")
        assert tokens == ["1.5", "+", "2.5"]

    def test_parse_with_whitespace(self):
        """Test that parse handles extra whitespace."""
        calc = Calculator()
        tokens = calc.parse("  2   +   3  ")
        assert tokens == ["2", "+", "3"]

    def test_parse_empty_expression(self):
        """Test that parse raises ValueError for empty expression."""
        calc = Calculator()
        with pytest.raises(ValueError, match="Empty expression"):
            calc.parse("")

    def test_parse_only_spaces(self):
        """Test that parse raises ValueError for whitespace-only expression."""
        calc = Calculator()
        with pytest.raises(ValueError, match="Empty expression"):
            calc.parse("   ")

    def test_parse_invalid_operand(self):
        """Test that parse raises ValueError for invalid operand."""
        calc = Calculator()
        with pytest.raises(ValueError, match="Invalid operand"):
            calc.parse("2 + abc")

    def test_parse_invalid_operator(self):
        """Test that parse raises ValueError for invalid operator."""
        calc = Calculator()
        with pytest.raises(ValueError, match="Invalid operator"):
            calc.parse("2 % 3")

    def test_parse_invalid_structure(self):
        """Test that parse raises ValueError for invalid structure."""
        calc = Calculator()
        with pytest.raises(ValueError, match="Invalid operand"):
            calc.parse("2 + + 3")

    def test_evaluate_addition(self):
        """Test that evaluate handles addition."""
        calc = Calculator()
        tokens = ["2", "+", "3"]
        result = calc.evaluate(tokens)
        assert result == 5.0

    def test_evaluate_subtraction(self):
        """Test that evaluate handles subtraction."""
        calc = Calculator()
        tokens = ["10", "-", "4"]
        result = calc.evaluate(tokens)
        assert result == 6.0

    def test_evaluate_multiplication(self):
        """Test that evaluate handles multiplication."""
        calc = Calculator()
        tokens = ["3", "*", "7"]
        result = calc.evaluate(tokens)
        assert result == 21.0

    def test_evaluate_division(self):
        """Test that evaluate handles division."""
        calc = Calculator()
        tokens = ["10", "/", "3"]
        result = calc.evaluate(tokens)
        assert abs(result - 3.333333333333) < 0.0001

    def test_evaluate_chained_operations(self):
        """Test that evaluate handles chained operations left-to-right."""
        calc = Calculator()
        tokens = ["10", "+", "5", "*", "2"]
        result = calc.evaluate(tokens)
        assert result == 30.0

    def test_evaluate_float_operands(self):
        """Test that evaluate handles float operands."""
        calc = Calculator()
        tokens = ["1.5", "+", "2.5"]
        result = calc.evaluate(tokens)
        assert result == 4.0

    def test_evaluate_division_by_zero(self):
        """Test that evaluate raises ValueError for division by zero."""
        calc = Calculator()
        tokens = ["5", "/", "0"]
        with pytest.raises(ValueError, match="Division by zero"):
            calc.evaluate(tokens)

    def test_evaluate_invalid_operand(self):
        """Test that evaluate raises ValueError for invalid operand."""
        calc = Calculator()
        tokens = ["2", "+", "abc"]
        with pytest.raises(ValueError, match="Invalid operand"):
            calc.evaluate(tokens)

    def test_evaluate_invalid_operator(self):
        """Test that evaluate raises ValueError for invalid operator."""
        calc = Calculator()
        tokens = ["2", "%", "3"]
        with pytest.raises(ValueError, match="Invalid operator"):
            calc.evaluate(tokens)

    def test_evaluate_invalid_structure(self):
        """Test that evaluate raises ValueError for invalid structure."""
        calc = Calculator()
        tokens = ["2", "+", "+", "3"]
        with pytest.raises(ValueError, match="Invalid expression structure"):
            calc.evaluate(tokens)

    def test_calculate_wrapper_delegates_to_class(self):
        """Test that calculate function delegates to Calculator class."""
        calc = Calculator()
        tokens = calc.parse("2 + 3")
        result = calculate("2 + 3")
        assert result == calc.evaluate(tokens)

    def test_calculate_with_floats(self):
        """Test that calculate function handles floats correctly."""
        calc = Calculator()
        tokens = calc.parse("1.5 + 2.5")
        result = calculate("1.5 + 2.5")
        assert result == calc.evaluate(tokens)

    def test_calculate_with_single_number(self):
        """Test that calculate function handles single numbers."""
        calc = Calculator()
        tokens = calc.parse("42")
        result = calculate("42")
        assert result == calc.evaluate(tokens)

    def test_calculate_with_single_float(self):
        """Test that calculate function handles single floats."""
        calc = Calculator()
        tokens = calc.parse("3.14")
        result = calculate("3.14")
        assert result == calc.evaluate(tokens)

    def test_calculate_division_by_zero(self):
        """Test that calculate function raises ValueError for division by zero."""
        with pytest.raises(ValueError, match="Division by zero"):
            calculate("5 / 0")

    def test_calculate_invalid_operand(self):
        """Test that calculate function raises ValueError for invalid operand."""
        with pytest.raises(ValueError, match="Invalid operand"):
            calculate("2 + abc")

    def test_calculate_invalid_operator(self):
        """Test that calculate function raises ValueError for invalid operator."""
        with pytest.raises(ValueError, match="Invalid operator"):
            calculate("2 % 3")

    def test_calculate_empty_expression(self):
        """Test that calculate function raises ValueError for empty expression."""
        with pytest.raises(ValueError, match="Empty expression"):
            calculate("")

    def test_calculate_only_spaces(self):
        """Test that calculate function raises ValueError for whitespace-only expression."""
        with pytest.raises(ValueError, match="Empty expression"):
            calculate("   ")

    def test_parse_single_number(self):
        """New Test: Ensure parse handles a single number correctly."""
        calc = Calculator()
        tokens = calc.parse("42")
        assert tokens == ["42"]

    def test_evaluate_single_number(self):
        """New Test: Ensure evaluate handles a single token list correctly."""
        calc = Calculator()
        result = calc.evaluate(["42"])
        assert result == 42.0
