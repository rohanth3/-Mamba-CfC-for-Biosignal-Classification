with open('tests/test_data.py', 'r') as f:
    text = f.read()

text = text.replace('f"S{sub}_A1_E{ex}.mat"', 'f"S{int(sub)}_A1_E{int(ex)}.mat"')

with open('tests/test_data.py', 'w') as f:
    f.write(text)
