import trl
print(f"TRL version: {trl.__version__}")

# Check SFTTrainer signature
import inspect
sig = inspect.signature(trl.SFTTrainer.__init__)
print("\nSFTTrainer parameters:")
for param in sig.parameters:
    if param != 'self':
        print(f"  - {param}")