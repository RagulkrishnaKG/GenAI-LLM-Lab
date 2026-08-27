# Experiment 07 - Output

## Sample Output
```python
Generated Function:
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

Debug Suggestion:
def factorial_fixed(n):
    result = 1
    for i in range(1, n + 1):
        result = result * i
    return result
```

## Result
An AI-powered assistant was successfully developed that generates Python code from natural-language instructions and identifies/fixes bugs in existing code using a pre-trained code LLM.
