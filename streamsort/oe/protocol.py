"""Decoder -> worker messages over multiprocessing.connection (localhost, stdlib, pickle).

decoder -> worker
  ("hello", probe, sample_rate, unit_ids, bank_mtime)
  ("buf",   probe, head, frames, units)        every buffer; head = samples received so far
  ("pulse", probe, onset_sample, ttl_index, t_arrival)  ttl_index None without a stim TTL on the stream;
                                               t_arrival = perf_counter when the onset block arrived
  ("bye",   probe)
worker -> decoder
  ("condition", key)                           condition of the next pulse (artifact suppression)
"""
DEFAULT_ADDRESS = ("127.0.0.1", 6010)
DEFAULT_AUTHKEY = b"streamsort"


def address(cfg: dict):
    r"""[host, port] (TCP), a socket path (AF_UNIX) or r"\\.\pipe\name" (Windows named pipe)."""
    a = cfg.get("worker_address", DEFAULT_ADDRESS)
    if not a:
        return None
    return a if isinstance(a, str) else (a[0], int(a[1]))


def family(cfg: dict):
    a = address(cfg)
    if isinstance(a, str):
        return "AF_PIPE" if a.startswith("\\\\") else "AF_UNIX"
    return "AF_INET"


def authkey(cfg: dict) -> bytes:
    k = cfg.get("authkey", DEFAULT_AUTHKEY)
    return k.encode() if isinstance(k, str) else k
