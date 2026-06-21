from pathlib import Path

import pyperclip

url0 = "https://github.com/NeuroAI-Research/neu-ai/blob/main/python"
root = Path("./src/neu_ai")
done = {}
names = {"dayan2005": "Theoretical Neuroscience book (Dayan, 2005)"}
out = []

for x in sorted(root.rglob("*.py")):
    if len(x.parts) == 4:
        p3, p4 = x.parts[2:]
        if p3 not in done:
            out.append(f"\n\n### {names.get(p3, p3.upper())}\n")
            done[p3] = True
        n_line = len(x.read_text().splitlines())
        if n_line:
            if p4.startswith(("m", "y")):
                p4 = p4[1:]
            p4 = p4.replace("_", " ")[:-3]
            out.append(f"- [{p4} ({n_line} lines)]({url0}/{x})")

print("\n".join(out))
pyperclip.copy("\n".join(out))
print("copied")
