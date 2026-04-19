
class ProcessingElement:
    """This is a PE (Processing Element) in the systolic array. Each PE multiplies and accumulates products!"""

    def __init__(self):
        """Initialize with zero accumulator. We start from zero because we sum up all the products."""
        self.sum = 0  # We keep the accumulator simple - just call it sum
        # FIXME: Might need to reset this between operations, check later

    def compute(self, x, y):
        """This is what each PE does - multiply two numbers and add to running sum."""
        product = x * y  # We compute the product
        self.sum += product  # We add it to our accumulator
        # Simple accumulation - this is the core of PE operations!
        return self.sum
