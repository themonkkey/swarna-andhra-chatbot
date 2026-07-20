"""Crawl apconstituencies.ap.gov.in for every Vision & Action Plan document.

The site's Documents API expects an AES-encrypted constituency id
(Encrypted_RequestDto), not a plain code. The encryption scheme was
recovered from the site's own public JS bundle (chunk-PEW76BJL.js):
AES-256-CBC, key = SHA256("12345678901234567890123456789012"),
fixed IV "1234567890123456", PKCS7 padding, base64-encoded ciphertext.
Verified against the site's own constituencyID values (matches exactly).

Downloads every referenced PDF, deduplicated by document id (state and
district-level PDFs repeat across many constituencies), into:
  corpus_files/vision_documents/state/
  corpus_files/vision_documents/district/<District>/
  corpus_files/vision_documents/constituency/<District>/<Constituency>/
  corpus_files/vision_documents/mandal/<District>/<Constituency>/
"""
import base64
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "corpus_files", "vision_documents")
CONSTITUENCIES_PATH = os.path.join(BASE_DIR, "constituencies_list.json")
MANIFEST_PATH = os.path.join(BASE_DIR, "corpus_files", "vision_documents_manifest.json")
API_BASE = "https://apconstituencies.ap.gov.in/CONST/api/Home"
FILE_BASE = "https://apconstituencies.ap.gov.in/CONST/filepath/"

_SECRET = "12345678901234567890123456789012"
_IV = "1234567890123456".encode()
_KEY = hashlib.sha256(_SECRET.encode()).digest()


def encrypt(plaintext):
    cipher = AES.new(_KEY, AES.MODE_CBC, _IV)
    ct = cipher.encrypt(pad(str(plaintext).encode(), AES.block_size))
    return base64.b64encode(ct).decode()


def safe_name(s):
    s = re.sub(r"\s+", "_", s.strip())
    return re.sub(r"[^\w\-.]", "", s)


def classify(category, title):
    c = (category or "").lower()
    t = (title or "").lower()
    if "state" in c:
        return "state"
    if "district" in c:
        return "district"
    if "mandal" in c:
        return "mandal"
    if "profile" in c or "profile" in t:
        return "profile"
    return "constituency"


def fetch_documents_for(session, code):
    enc = encrypt(code)
    r = session.post(f"{API_BASE}/Documents", json={"encryptedConstId": enc}, timeout=30)
    if r.status_code != 200:
        return []
    try:
        return r.json() or []
    except Exception:
        return []


def download_file(session, file_path, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return "cached"
    url = FILE_BASE + file_path.lstrip("/")
    tmp_path = dest_path + ".part"
    try:
        r = session.get(url, timeout=120, stream=True)
        if r.status_code != 200:
            return f"http_{r.status_code}"
        expected = r.headers.get("Content-Length")
        expected = int(expected) if expected and expected.isdigit() else None

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        written = 0
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                written += len(chunk)

        # a stream cut short by a timeout/reset writes a truncated file that still
        # looks "cached" on the next run unless we verify it against the server's
        # declared size before ever letting it become the real file.
        if expected is not None and written != expected:
            os.remove(tmp_path)
            return f"error:truncated ({written}/{expected} bytes)"

        os.replace(tmp_path, dest_path)
        return "downloaded"
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return f"error:{e}"


WORKERS = 12
_thread_local = threading.local()
_seen_lock = threading.Lock()
_manifest_lock = threading.Lock()


def get_session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def claim(doc_id, seen_doc_ids):
    """Return True if this thread should download doc_id (first to claim it)."""
    with _seen_lock:
        if doc_id in seen_doc_ids:
            return False
        seen_doc_ids.add(doc_id)
        return True


def process_constituency(i, total, c, seen_doc_ids, manifest):
    code = c["code"]
    cname = c["name"]
    district = c["district"] or "Unknown"
    session = get_session()
    print(f"[{i}/{total}] {cname} ({district}) code={code}")

    docs = fetch_documents_for(session, code)
    if not docs:
        print(f"  ! {cname}: no documents returned")
        return

    for d in docs:
        doc_id = d["id"]
        kind = classify(d.get("category"), d.get("title"))
        ext = os.path.splitext(d["filePath"])[1].lower() or ".pdf"
        fname = safe_name(d["title"]) + ext

        if kind == "state":
            dest = os.path.join(OUT_DIR, "state", fname)
        elif kind == "district":
            dest = os.path.join(OUT_DIR, "district", safe_name(district), fname)
        elif kind == "mandal":
            dest = os.path.join(OUT_DIR, "mandal", safe_name(district), safe_name(cname), fname)
        else:
            dest = os.path.join(OUT_DIR, "constituency", safe_name(district), safe_name(cname), fname)

        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            status = "cached"
            claim(doc_id, seen_doc_ids)
        elif claim(doc_id, seen_doc_ids):
            status = download_file(session, d["filePath"], dest)
        else:
            status = "dup_skipped"

        with _manifest_lock:
            manifest.append({
                "doc_id": doc_id, "constituency": cname, "district": district,
                "title": d["title"], "category": d.get("category"), "kind": kind,
                "fileSizeMB": d.get("fileSizeMB"), "isActive": d.get("isActive"),
                "dest": os.path.relpath(dest, BASE_DIR), "status": status,
            })
        print(f"    {status:12s} {cname}: {d['title']}")


def main():
    with open(CONSTITUENCIES_PATH) as f:
        constituencies = json.load(f)

    print(f"Crawling documents for {len(constituencies)} constituencies with {WORKERS} workers...")
    seen_doc_ids = set()
    manifest = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [
            pool.submit(process_constituency, i, len(constituencies), c, seen_doc_ids, manifest)
            for i, c in enumerate(constituencies, 1)
        ]
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                print(f"  ! worker error: {exc}")

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=1)

    downloaded = sum(1 for m in manifest if m["status"] == "downloaded")
    cached = sum(1 for m in manifest if m["status"] == "cached")
    dups = sum(1 for m in manifest if m["status"] == "dup_skipped")
    errs = sum(1 for m in manifest if m["status"].startswith(("error", "http_")))
    print(f"\nDone. {downloaded} downloaded, {cached} already cached, {dups} duplicates skipped, {errs} errors.")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
