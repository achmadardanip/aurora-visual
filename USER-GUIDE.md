# AURORA User Guide (easy English)

This guide shows you, step by step, how to start the AURORA system and check
whether a caption matches an image.

**The big picture.** AURORA has three apps that work as one team:

| App | Address | Job |
| --- | --- | --- |
| 1 · Visual | http://127.0.0.1:5171 | Reads the image, splits the caption into small claims, checks each claim against what is visible. |
| 2 · Evidence | http://127.0.0.1:5172 | Searches a fact-check corpus for related articles and checks whether the image/text looks AI-made. |
| 3 · Decision | http://127.0.0.1:5173 | Combines everything into one verdict per claim, with quotes and sources you can click. |

**Start here if you are in a hurry:** run `make system`, open
http://127.0.0.1:5173, click **“Muat fixture demo”**, then **“Jalankan fusi”**.
You will see a full verdict in under a minute — no keys, no setup.

---

## 1. Before you start: one-time setup

You only need to do this once on a new computer.

### 1.1 Install the tools

You need: **Python 3.12** (with `uv`), **Node.js 20+** (with `npm`), and
**Tesseract OCR** (for reading text inside images).

```sh
make setup
```

This installs Python packages, installs the three frontends, and prepares
the databases. It takes a few minutes the first time.

### 1.2 Which settings you must configure (and which you can skip)

Everything already works out of the box **except** the paid/external AI
services. This is the current state on this machine:

| Setting | Where | Status | Do you need to do anything? |
| --- | --- | --- | --- |
| Visual AI (DeepSeek via Modal) | Module 1 → Settings | ✅ Already set | No |
| AI-image detection (Hive, module 1) | Module 1 → Settings | ✅ Already set | No |
| Local fact-check search (module 2) | — | ✅ Always works | No |
| Web search (Tavily, module 2) | `aurora-evidence/.env.evidence` | ❌ Optional | Only if you want Google-style web results |
| AI-text detection (GPTZero, module 2) | `aurora-evidence/.env.evidence` | ❌ Optional | Only if you want an AI-writing score |
| AI-image detection (Hive, module 2) | `aurora-evidence/.env.evidence` | ❌ Optional | Only if you want a second detector |
| Decision settings (module 3) | — | ✅ Defaults work | No |

**In short: you can skip configuration completely.** The optional services
only add extra evidence; the system is fully usable without them.

### 1.3 (Optional) Turn on web search or AI detectors for module 2

Only do this if you have the API keys.

1. Copy the example file:
   ```sh
   cp aurora-evidence/.env.evidence.example aurora-evidence/.env.evidence
   ```
2. Open `aurora-evidence/.env.evidence` in a text editor and paste your key
   after the `=` sign, for example:
   ```
   TAVILY_API_KEY=tvly-your-key-here
   ```
3. Save the file and restart the system (see step 2). On the module 2
   settings page the provider will change from “Not available” to “Ready”.

Never put API keys anywhere else (not in the browser, not in git). The
`.env.evidence` file is ignored by git.

---

## 2. Start the app

Open a terminal in the project folder and run:

```sh
make system
```

Wait until you see “AURORA system is running”. Then open these pages:

- http://127.0.0.1:5171 — Visual (module 1)
- http://127.0.0.1:5172 — Evidence (module 2)
- http://127.0.0.1:5173 — Decision (module 3, **start here for the full check**)

To stop everything, press `Ctrl+C` in that terminal.

> If a port is already in use (for example after a crash), stop leftovers
> first: `pkill -f "scripts/system.py"; pkill -f uvicorn; pkill -f vite`,
> then run `make system` again.

---

## 3. The 5-minute tour (demo, no keys needed)

1. Open **http://127.0.0.1:5173** (Decision app).
2. Click **“Muat fixture demo”**. This loads a small made-up example:
   2 claims, 3 evidence documents (one of them a deliberate duplicate).
3. Click **“Jalankan fusi”**. The app verifies each claim against each
   document. Wait a few seconds.
4. Read the **verdict box** at the top. It says one of:
   - **Didukung** (Supported) — evidence backs the claim.
   - **Bertentangan** (Contradicted) — evidence contradicts it.
   - **Bukti tidak cukup** (Not enough evidence) — the app refuses to
     decide. This is normal and honest: without matching calibration data
     the system abstains instead of guessing.
5. Click a claim row to see **why**: the exact quote, the publisher, the
   date, and which independent source group it belongs to. Quotes are always
   exact copies from the document — never reworded.
6. Try **“Simpan tinjauan”** (human review): write your name, pick your own
   verdict, give a reason. Your review is saved *next to* the model output —
   it never overwrites it.
7. Try the **MD / HTML / CSV / ZIP** buttons to download the full report.

---

## 4. Check a real caption + image (the full pipeline)

1. Open **http://127.0.0.1:5173** and click **“Import bundle”**,
   or start from scratch:
2. In the fuse page, upload an image (JPG/PNG/WebP) and type the caption
   you want to check. Example:
   > “Bidang ini berwarna merah” (This field is red)
3. Click **“Pipeline penuh”**. The app now calls the other two services
   automatically:
   - **Module 1** splits your caption into small checkable claims and
     checks each one against the image (red/green counts below).
   - **Module 2** searches the fact-check corpus for related articles and
     runs AI-content checks.
   - **Module 3** fuses everything into the verdict you see.
4. Wait for “Fusi selesai”. If a step fails (for example a provider is
   down), the app still gives you a **partial** result with a visible
   warning instead of pretending everything worked.

### How to read the result

- **Label dasar (base label)** — what the model thinks before the safety
  rules: Supported / Contradicted / InsufficientEvidence.
- **Status / Putusan (verdict)** — the operational answer after safety
  rules. If it says **Bukti tidak cukup** with reason `UNCALIBRATED`, it
  means: “I have an opinion, but no matching calibration data, so I
  refuse to decide.” This is the correct behavior.
- **Set keyakinan (confidence set)** — the list of labels the calibration
  still allows. `null` means “not calibrated”.
- **Tautan bukti (evidence links)** — click them: each has an exact quote,
  a stance (Supports / Contradicts / Unclear), the publisher, and the date.
- **Duplicate cluster / independence group** — ten copies of one story on
  ten websites count as **one** vote, not ten.

### Useful rules of thumb

- An **AI-content score** is about *who made the pixels*, never about
  whether the caption is true. A real photo can carry a false caption, and
  an AI image can carry a true one. The app never mixes these up.
- **Tidak teramati (Unobservable)** in module 1 means “cannot be judged
  from this image” — not “false”.
- Dates the app cannot find stay **unknown**; the app never guesses them.

---

## 5. Using each app on its own

- **Module 1 (5171)** — upload image(s), paste the caption, choose local or
  AI reading, click **“Jalankan analisis”**. Click a claim point to see the
  highlighted image regions. Use **“Koreksi poin klaim”** to fix a claim and
  re-run.
- **Module 2 (5172)** — type a claim, optionally upload the image, click
  **“Jalankan pencarian”**. Tabs: **Bukti** (articles with quotes),
  **Asal Gambar** (local image matches), **Deteksi AI** (image vs text
  panels), **Jejak Pencarian** (every query, timing, provider status).
  Filter by kind, sort by relevance or date, export CSV/ZIP.
- **Module 3 (5173)** — import a bundle from apps 1/2, or use **“Pipeline
  penuh”** to run everything. Calibrate on the Settings page once you have
  a calibration artifact (see `aurora-decision/docs/`).

---

## 6. If something goes wrong

| Symptom | What to do |
| --- | --- |
| `make system` says a port is in use | Kill leftovers (step 2 note) and start again. |
| Job says “failed” with a `detail` code | The detail (e.g. `http_error_402`) names the real cause; quota/auth errors come from the provider account, not the app. |
| Module 2 shows “Tidak tersedia” for Tavily/GPTZero/Hive | Normal without keys — local corpus search still works. Add keys (step 1.3) to enable them. |
| Verdict is always “Bukti tidak cukup” | Expected until you calibrate (atom-level) — the app abstains rather than guess. Base labels still show the model’s opinion. |
| UI looks outdated after an update | Hard-refresh the browser page (the dev server hot-reloads code). |

Configuration values live in `var/settings.json` (module 1),
`var-evidence/settings.json` (module 2), `var-decision/settings.json`
(module 3). Secrets there are never sent to git. Deleting those folders
resets an app to defaults (your cases will be gone too).
