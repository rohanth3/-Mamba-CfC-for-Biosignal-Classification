import ast
import codecs
from datetime import datetime

with open('tests/test_data.py', 'r') as f:
    tree = ast.parse(f.read())
    
docstrings = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
        doc = ast.get_docstring(node)
        if doc:
            docstrings[node.name] = doc.split('\n')[0]

with codecs.open('reports/test_results.txt', 'r', 'utf-16le') as f:
    pytest_output = f.read()

report = f'''Test Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Explanation of missing tests:
- `test_metadata_and_leakage` and `test_stats_reproduce_val_test` from the initial suite were combined and significantly upgraded into `test_shapes_labels_disjoint` and `test_data_integrity` to enforce stronger and more exact checks (like exact start index sequences against actual raw files and full scaling reproducibility) rather than just partial logic.

Test Assertions:
'''
for name, doc in docstrings.items():
    report += f'- {name}: {doc}\n'

report += '\nPytest Output:\n' + pytest_output

# Because redirection from python in powershell uses utf-16, let's write it in standard utf-8
with open('reports/test_results.txt', 'w', encoding='utf-8') as f:
    f.write(report)
