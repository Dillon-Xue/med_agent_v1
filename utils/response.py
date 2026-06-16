# utils/response.py

import time
import uuid

def build_response(answer, source, debug=None, trace=None, success=True):
    return {
        "success": success,
        "id": str(uuid.uuid4()),
        "source": source,
        "answer": answer,
        "debug": debug or {},
        "trace": trace or [],
        "timestamp": time.time()
    }
