class CustomHTTPException(Exception):
    def __init__(self, status_code: int, code: str, message: str, detail: dict = None):
        self.status_code = status_code
        self.message = message
        self.data = {
            "code": code,
            "detail": detail
        } 