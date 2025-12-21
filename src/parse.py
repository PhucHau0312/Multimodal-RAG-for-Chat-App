import re 

def extract_sections(text):
    """
    Extract sections from the generated text.
    
    Args:
        text (str): The generated text
        
    Returns:
        dict: Extracted sections
    """
    sections = {}
    headings = ["Evidence", "Chain of Thought", "Answer"]
    
    for i in range(len(headings)):
        heading = headings[i]
        next_heading = headings[i + 1] if i + 1 < len(headings) else None
        
        if next_heading:
            pattern = rf"## {heading}:(.*?)(?=## {next_heading}:)"
        else:
            pattern = rf"## {heading}:(.*)"
        
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[heading] = match.group(1).strip()
        else:
            sections[heading] = ""
    
    return sections


def parse_combined_output(output):
    """
    Parse the output of the combination step.
    
    Args:
        output (str): The combined output text
        
    Returns:
        dict: Parsed sections
    """
    sections = {'Analysis': '', 'Conclusion': '', 'Final Answer': ''}
    current_section = None

    for line in output.split('\n'):
        if line.startswith('## '):
            current_section = line[3:].strip(':')
        elif current_section and current_section in sections:
            sections[current_section] += line + '\n'

    # Clean up the sections
    for key in sections:
        sections[key] = sections[key].strip()

    return sections