
def make_response(status: str, message: str, data: dict = None, code: int = 200):
    return {
        "status": status,
        "message": message,
        "data": data or {}
    }, code
