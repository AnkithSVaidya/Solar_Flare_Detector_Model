"""Download the SDOBenchmark dataset via kagglehub.

Handles the intercepting-proxy SSL quirk seen in some corporate networks.
Prints the local path the dataset was extracted to.
"""
import os
import ssl
import warnings

# Tolerate an intercepting corporate proxy whose CA chain doesn't verify.
os.environ.setdefault("PYTHONHTTPSVERIFY", "0")
ssl._create_default_https_context = ssl._create_unverified_context

# kagglehub uses `requests`, which ignores the ssl default context above and
# uses certifi instead. Force its sessions to skip verification of the proxy
# cert (and silence the resulting InsecureRequestWarning).
try:
    import requests
    from urllib3.exceptions import InsecureRequestWarning

    warnings.simplefilter("ignore", InsecureRequestWarning)

    _orig_request = requests.Session.request
    _orig_send = requests.Session.send
    _orig_merge = requests.Session.merge_environment_settings

    def _no_verify_request(self, *a, **k):
        k["verify"] = False
        return _orig_request(self, *a, **k)

    def _no_verify_send(self, *a, **k):
        k["verify"] = False
        return _orig_send(self, *a, **k)

    def _no_verify_merge(self, *a, **k):
        settings = _orig_merge(self, *a, **k)
        settings["verify"] = False
        return settings

    requests.Session.request = _no_verify_request
    requests.Session.send = _no_verify_send
    requests.Session.merge_environment_settings = _no_verify_merge
except Exception as _e:  # noqa: BLE001
    print("requests patch skipped:", repr(_e), flush=True)

import kagglehub

print("kagglehub", kagglehub.__version__, flush=True)
try:
    path = kagglehub.dataset_download("fhnw-i4ds/sdobenchmark")
    print("DOWNLOADED_TO:", path, flush=True)
except Exception as e:  # noqa: BLE001
    print("DOWNLOAD_ERROR:", repr(e), flush=True)
