# Coding rules
1. Write technical, direct answers and do not waste time.
2. Do not write comments unless they are genuinely necessary.
3. In Python, use exactly one blank line between classes, methods, and functions.
4. Use blank lines inside Python functions and methods to separate logical steps visually, without adding comments.
5. Always try to delete code and simplify the codebase.
6. Prefer a few direct functions over many tiny helpers, wrappers, or dynamic abstractions.
7. Keep variables short-lived and avoid naming intermediate values unless they make the code easier to read.
8. Do not add classes unless they hold real state or remove meaningful complexity.
9. Keep CLI options minimal, but every option must have a practical default when possible.
10. When optimizing, preserve a simple data flow that can be read from top to bottom.
11. Avoid bloat: do not introduce abstractions, helpers, comments, types, options, or files that are not pulling their weight.
12. After finishing code changes, run `uvx ruff check .` and `uvx ty check .` to catch linting and typing issues.
