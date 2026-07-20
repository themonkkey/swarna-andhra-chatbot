# Swarna Andhra — Session Handoff (2026-07-17)

For next Claude session. State of the vision-documents work.

## Corpus status — COMPLETE
- Location: `~/swarna-andhra-chatbot/corpus_files/vision_documents/`
- 1,357 PDFs, 2.9 GB. Structure: `state/` (2), `district/` (26), `constituency/<District>/<Constituency>/` (360), `mandal/<District>/<Constituency>/` (951)
- Manifest: `corpus_files/vision_documents_manifest.json` — 1,656 entries (state + district docs repeat across constituencies, deduplicated on disk → 1,357 files). 0 errors.
- Crawler: `scripts/crawl_vision_documents.py` (venv: `./venv/bin/python`). Idempotent — rerun anytime to pick up new portal docs. Last rerun 2026-07-17: 0 new, 1,656 cached, 0 errors.
- Includes new "-2026" mandal plans dated 14/7/2026 (Rampachodavaram: 11 old + 11 new = 22 files).

## Missing district plans — RESOLVED (they don't exist)
- `district/` has 26 plans; AP now has 28 districts. **Markapuram** and **Polavaram** district plans are NOT published anywhere (verified 2026-07-17 via portal API probe + district sites + APSDPS + Telugu/English web sweep, 15-agent search).
- Reason: plans written Dec 2024–Jan 2025 for then-26 districts; Markapuram & Polavaram became districts 01/01/2026 (notified 31/12/2025).
- Territory coverage: Markapuram → `district/Prakasam/Prakasam_District_Vision_Action_Plan.pdf`; Polavaram → `district/Alluri_Seetharama_Raju/Alluri_Sitharama_Raju_District_Vision_Action_Plan.pdf`.
- Watch for future publication at: `markapuram.ap.gov.in/document-category/plan-report/`, `polavaram.ap.gov.in/documents/`, and apconstituencies.ap.gov.in (then rerun crawler).

## Google Drive upload — IN PROGRESS
- Command: `rclone copy ~/swarna-andhra-chatbot/corpus_files/vision_documents/ "gdrive:Swarna_Andhra_Vision_Documents" --exclude ".DS_Store" --transfers 8`
- rclone installed via Homebrew; remote `gdrive` authorized (OAuth token in `~/.config/rclone/rclone.conf`). Note: rclone's shared client_id retiring during 2026 — make own client_id eventually.
- Was running in background at session end. Verify with: `rclone size "gdrive:Swarna_Andhra_Vision_Documents"` — should show 1,356 files (~2.9 GB, .DS_Store excluded). If incomplete, rerun the copy command — it resumes/skips existing.

## Portal technical notes
- Site: apconstituencies.ap.gov.in (SPA; direct curl of routes → 404, sometimes 502)
- API: POST `https://apconstituencies.ap.gov.in/CONST/api/Home/Documents` body `{"encryptedConstId": ENC}`; ENC = AES-256-CBC(constituency code), key=SHA256("12345678901234567890123456789012"), IV="1234567890123456", PKCS7, base64
- Files: `https://apconstituencies.ap.gov.in/CONST/filepath/<filePath>`
- Constituency codes: `constituencies_list.json`
