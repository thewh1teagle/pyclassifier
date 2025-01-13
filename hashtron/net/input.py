class Input:
    def __init__(self, obj):
        # Check if the wrapped object has a 'feature' method
        if not hasattr(obj, 'feature') or not callable(obj.feature):
            raise ValueError("The wrapped object must have a 'feature' method.")
        self.obj = obj

    def feature(self, n: int) -> int:
        # Call the wrapped object's feature method
        result = self.obj.feature(n)
        
        # Check if the wrapped object has a 'parity' method
        if hasattr(self.obj, 'parity') and callable(self.obj.parity):
            # If it does, XOR the result with the parity
            result ^= self.obj.parity()
        
        return result

    def parity(self) -> int:
        # Check if the wrapped object has a 'parity' method
        if hasattr(self.obj, 'parity') and callable(self.obj.parity):
            # If it does, return the parity
            return self.obj.parity()
        return 0
