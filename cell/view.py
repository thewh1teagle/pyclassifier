import json

class HashtronView:
    def __init__(self, hashtron):
        self.hashtron = hashtron
    
    def write_json(self) -> str:
        return json.dumps(self.hashtron.program)
    
    def read_json(self, b) -> None:
        self.hashtron.program = json.loads(b)
