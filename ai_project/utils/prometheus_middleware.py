from prometheus_client import Counter, Histogram
import time

REQUEST_COUNT = Counter("request_count", "Total requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency", ["method", "endpoint"])

async def prometheus_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(duration)
    return response