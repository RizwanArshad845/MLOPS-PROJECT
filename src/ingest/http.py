"""requests session with retries/backoff for flaky public APIs."""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TIMEOUT = 60


def session(headers: dict | None = None) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=6, backoff_factor=2, status_forcelist=(408, 429, 500, 502, 503, 504),
                  allowed_methods=("GET",), respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "smogcast-mlops/1.0", **(headers or {})})
    return s


def get_json(s: requests.Session, url: str, params: dict) -> dict:
    r = s.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()
