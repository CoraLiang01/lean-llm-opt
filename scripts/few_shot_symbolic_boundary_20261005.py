"""Keep mathematical structure while excluding data sections from code-generation input."""
import re


def symbolic_model_for_codegen(text):
    """Transfer variables, objectives and constraints, not echoed CSV tables/parameters."""
    lines, keep = [], False
    for line in text.splitlines():
        heading = re.match(r'^\s*(?:#{1,6}\s+(.+)|(?:\d+\.\s*)?\*\*([^*]+)\*\*\s*:?(.*))$', line)
        if heading:
            title = (heading.group(1) or heading.group(2)).casefold()
            if any(word in title for word in ('parameter', 'data mapping', 'data source', 'given data', 'given:')):
                keep = False
            elif any(word in title for word in ('variable', 'objective', 'constraint', 'domain')):
                keep = True
        if keep and not re.match(r'^\s*\|.*\|\s*$', line):
            lines.append(line)
    result = '\n'.join(lines).strip()
    if not result or not re.search(r'objective|(?:\\min|\\max)|minimize|maximize', result, re.I):
        raise ValueError('No recognizable symbolic objective/constraints for code generation; no data or regenerated model is substituted')
    return result
