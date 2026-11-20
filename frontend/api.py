"""Small authenticated HTTP client for Streamlit-to-FastAPI calls."""
import os
import requests


class APIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, payload=None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload
        self.request_id = (payload or {}).get("request_id") if isinstance(payload, dict) else None


class AuraAPI:
    def __init__(self, token: str | None = None, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("AURA_API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.token = token

    def request(self, method: str, path: str, *, params=None, json=None, files=None, timeout=12, raw=False):
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            response = requests.request(method, f"{self.base_url}{path}", params=params, json=json,
                                        files=files, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            raise APIError(f"AURA API is unavailable: {exc}") from exc
        if not response.ok:
            error_payload=None
            try:
                error_payload=response.json()
                detail = error_payload.get("detail", "Request failed")
                if isinstance(detail, dict): detail = detail.get("message", "Request validation failed")
                if not isinstance(detail, str): detail = "Request validation failed"
            except (ValueError, AttributeError):
                detail = "Request failed"
            request_id = response.headers.get("X-Request-ID")
            if not request_id and isinstance(error_payload, dict): request_id = error_payload.get("request_id")
            if request_id: detail = f"{detail} (request ID: {request_id})"
            raise APIError(detail, response.status_code, error_payload)
        if raw: return response.content
        if response.content:
            try: return response.json()
            except ValueError: return response.text
        return None

    def get(self, path, **kwargs): return self.request("GET", path, **kwargs)
    def post(self, path, **kwargs): return self.request("POST", path, **kwargs)
    def put(self, path, **kwargs): return self.request("PUT", path, **kwargs)
