def parse_csv(text: str) -> list[list[str]]:
    """
    Parses a CSV string following RFC 4180 rules.
    
    Args:
        text: The CSV content to parse.
        
    Returns:
        A list of rows, where each row is a list of field strings.
        
    Raises:
        ValueError: If there's an unclosed quote in the input.
    """
    if not text:
        return []

    rows = []
    current_row = []
    current_field = []
    in_quotes = False
    i = 0
    n = len(text)

    while i < n:
        char = text[i]

        if in_quotes:
            if char == '"':
                # Check for escaped quote ("")
                if i + 1 < n and text[i + 1] == '"':
                    current_field.append('"')
                    i += 1  # Skip the second quote
                else:
                    # End of quoted field
                    in_quotes = False
            else:
                current_field.append(char)
        else:
            if char == '"':
                # Start of quoted field
                # Note: RFC 4180 says quotes must be at the start of a field.
                # For simplicity and based on ACs, we'll handle them as they appear.
                in_quotes = True
            elif char == ',':
                # End of field
                current_row.append("".join(current_field))
                current_field = []
            elif char == '\n':
                # End of row (LF)
                current_row.append("".join(current_field))
                rows.append(current_row)
                current_row = []
                current_field = []
            elif char == '\r':
                # Handle CRLF
                if i + 1 < n and text[i+1] == '\n':
                    current_row.append("".join(current_field))
                    rows.append(current_row)
                    current_row = []
                    current_field = []
                    i += 1 # Skip the \n
                else:
                    # Treat lone \r as a newline if it's not followed by \n?
                    # RFC 4180 specifies CRLF. Let's treat it like LF for robustness,
                    # but usually we just follow AC6 which explicitly mentions CRLF.
                    current_row.append("".join(current_field))
                    rows.append(current_row)
                    current_row = []
                    current_field = []
            else:
                current_field.append(char)
        i += 1

    if in_quotes:
        raise ValueError("Unclosed quote encountered")

    # Handle the last field/row if text doesn't end with a newline
    # But we must be careful about AC 10 (trailing newline -> no empty row)
    
    # If the loop ended and we have content or were in the middle of a row,
    # but the very last character was NOT a newline, we should add the final record.
    
    # Let's check if the text ended with a newline.
    if n > 0 and (text[-1] == '\n' or (text[-2:] == '\r\n' if n >= 2 else False)):
        # If it ends with a newline, the last field/row was already pushed by the \n logic.
        # UNLESS we had an empty line at the end? 
        # AC 10: "a,b\n" -> [["a", "b"]] - this is handled because when i reached \n, it appended row and reset.
        pass
    else:
        # No trailing newline, so we must push the current state as a final row
        current_row.append("".join(current_field))
        rows.append(current_row)

    return rows
