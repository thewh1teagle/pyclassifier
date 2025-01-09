import zlib
import json

class FeedforwardNetworkIO:
    def __init__(self, net):
        self.net = net

    def read_zlib_weights_from_file(self, file_name: str) -> bool:
        with open(file_name,'rb') as f:
            file_content=zlib.decompress(f.read())
        if len(file_content) <= 2:
            # empty model or uknown format
            return True
        if file_content[0] != ord('['):
            raise ValueError("unknown model format: please fix your model, or update pyclassifier (pip is hashtron) to recognize this model")
        decoded = file_content.decode('ascii')
        parsed = json.loads(decoded)
        i = 0
        for layer in self.net.network.layers:
            for cell in layer:
                cell.view.read_json(json.dumps(parsed[i]))
                i += 1
        return i == len(parsed)
