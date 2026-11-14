# Ashline

Ashline is the family group chat that remembers your home. After a wildfire, relatives text old
photos, Ashline circles and prices what it sees, calls to help recall what no photo shows, and
builds the evidence-linked contents list an insurer asks for.

This repo currently holds the **worker** side of the **home twin seed**: given the survivor's
address, it pulls the Matterport 3D scan (inside), the Vexcel aerial (outside) and the CoStar sale
listing (the record) of that home and turns them into the first batch of evidence for the case.

The three sources are used together. For example: the listing says there is a 2-car garage, the
aerial from before the fire shows a second building on the lot, and the scan never went inside it,
so Ashline asks "What was kept there: tools, bikes, storage?"

Everything the seed finds is a **suggestion**. A listing scan can show staging or a previous
owner's furniture, so nothing becomes an inventory item until the family confirms it.

## Setup

```bash
cd worker
python -m venv .venv && source .venv/bin/activate
pip install -e '.[agent,claude,dev]'     # add ,local-vision to run item detection without an API key
cp ../.env.example ../.env               # then fill in the keys
pytest -q
```

Keys come from `/costar-keys` in `#costar-group` on the HackPrinceton Discord (a teammate has to
run it). `.env` is git-ignored; nothing in this package prints or stores a key.

## Use

```bash
ashline lookup "12 Example Ave, Los Angeles, CA 90001"   # is there a scan for this address?
ashline seed --case demo                                 # home from ASHLINE_DEMO_SID in .env
ashline seed --case fam1 --address "12 Example Ave, Los Angeles, CA 90001" --loss-date 2025-01-07
ashline questions --case demo                            # voice-call questions from the gaps
ashline export --case demo --include-suggested           # cases/demo/inventory.csv
ashline kit-info                                         # what is inside the CoStar kit
ashline listing --sid <scan id>                          # the CoStar listing behind a scan
```

With the CoStar kit configured (`COSTAR_KIT_ZIP` or `COSTAR_KIT_URL`, plus `COSTAR_ZIP_PASSWORD`), the
seed also reads the listing: beds, baths, size and year built go into the family message, and what
the listing mentions but the scan does not show (a garage, a basement, a washer and dryer, named
appliances) becomes questions. Without the kit that step is skipped.

`seed` is safe to run again: scan photos, detections and aerial results are cached per case, and
answers the family already gave are kept. Run it once before the demo to precompute the demo home.

## Talking to Ashline

The conversation lives in `worker/ashline/chat.py` and does not depend on the messaging channel. A
bridge posts each incoming message to `POST /inbound` and sends the replies it gets back; `POST /tick`
(or the worker's own timer, when `ASHLINE_BRIDGE_URL` is set) produces the messages that come later:
the digest after a burst of photos, and the owner's daily review.

To try it without any messaging service:

```bash
ashline chat --as +15550001 --name Mom
```

```text
hi                               -> names the home, asks the address, the rooms, gives an invite code
/as +15550002 Maya               -> become a relative
HOME-1234                        -> join with the code
/photo ~/Pictures/thanksgiving.jpg
/wait                            -> the quiet window passes; the annotated photo comes back
/group rivera                    -> talk in the family group chat (Ashline stays quiet unless asked)
```

What it does: onboarding in a few messages; relatives join with a code; one acknowledgement and one
digest per burst of photos; a single photo comes back annotated; repeats are merged and a borderline
match becomes one "same item?" question; commands (`list`, `how much so far`, `show photo 3`,
`review`, `OK`, `remove 4`, `undo`, `export`, `pause`, `call me`, `delete my case`); a review for the
owner, one room at a time when a lot is waiting; and a memory conversation by text. In a group chat
it answers photos, messages addressed to it, replies to its own question and command words, and
otherwise says nothing.

If the address has a 3D scan, onboarding offers to start from it and asks when the fire was, so the
aerial photo comes from before it.

## Inventory page

```bash
cd worker && uvicorn ashline.web:app --port 8000
```

Each case has two long, unguessable links and no login: `/p/<owner_token>` for the survivor (confirm,
fix, remove, export) and `/p/<member_token>` for relatives (view only). The page is rendered from the
case store on every request, so it is always current. It shows the meter (how much of what SB 495
still requires is documented, split into evidence-backed, from memory and waiting for an OK), room
filters, an item grid with the crop from the original photo, and a detail sheet with every sighting
of the item. The owner can download the spreadsheet (XLSX or CSV) and the PDF for the insurer.

Prices are replacement-cost estimates with a range (`pricing.py`). The range comes from
`eval/calibration.json` once that has been measured; until then it is labelled a rough allowance
and no coverage is claimed.

## Tools

| Tool | Module | What it does |
| --- | --- | --- |
| `home_lookup` | `ashline/costar/index.py` | Address text or lat/lon to scan id, lat/lon, listing id and capture date. Returns "no scan" cleanly, and asks for the unit in multi-unit buildings. |
| `scan_rooms` | `ashline/costar/matterport.py` | Room list with dimensions, furnished flag and the photographer's capture notes. |
| `scan_photos` | `ashline/costar/matterport.py` | Picks about 10 of a scan's ~45 panoramas (big, valuable rooms first) and saves their four horizon faces, each tagged with scan, panorama, face and room. |
| `aerial_parcel` | `ashline/costar/vexcel.py` | Parcel, buildings, yard findings and an annotated aerial image from the last flight **before the loss date**. |
| `listing_record` | `ashline/costar/listing.py` | The CoStar listing behind a scan: scan id to listing ids through the kit's index CSV, then the full record and its features from the kit's JSONL files. |

Around them: `detect.py` (item detection, build spec 6.3), `dedupe.py` (scan priors for 6.4),
`gaps.py` (voice-call questions, 6.7), `export.py` (CSV with `source` and evidence date, 6.8),
`store.py` (SQLite case store, 6.9), `seed.py` (the whole seed) and `agent.py` (deepagents).

### Agent

`ashline.agent.build_agent()` returns a LangChain deep agent:

- **Subagents**: `inside` (scan rooms and photos) and `outside` (parcel and yard) each return a
  short summary to the orchestrator.
- **Human-in-the-loop**: `confirm_item`, `reject_item` and `answer_finding` are gated with
  `interrupt_on`, so they wait for a family reply.
- **Planning** and the **virtual filesystem** hold the per-case plan and one inventory note per room.

### Item detection

`ASHLINE_DETECTOR` picks how photos are read:

| Value | Needs | Notes |
| --- | --- | --- |
| `claude` | `ANTHROPIC_API_KEY` | Claude through the Anthropic SDK, with the item list constrained to a JSON schema. Model from `ASHLINE_CLAUDE_MODEL` (default `claude-opus-5-5`). |
| `vlm` | `ASHLINE_VLM_API_KEY` | Any OpenAI-compatible chat endpoint (`ASHLINE_VLM_BASE_URL`, `ASHLINE_VLM_MODEL`). |
| `florence` | nothing | Florence-2 run locally: about 15 seconds per photo on a laptop CPU, fewer small items found. |

Left unset, it uses `claude` when an Anthropic key is set, then `vlm`, then `florence`. Detections
are cached per case; `ashline seed --case demo --redetect` throws the cache away and runs the
current detector again.

### Merging repeats

Neighbouring panoramas see the same objects, so the same sofa is detected several times, in
different words. Detections are merged in three steps (`dedupe.py`, `merge_claude.py`):

1. An object cut by the seam between two faces of one panorama is joined back into one.
2. Detections of the same object from different panoramas of a room are grouped. With an Anthropic
   key this is one text-only request per room (`ASHLINE_MERGER=claude`); without one, word-matching
   rules are used (`ASHLINE_MERGER=rules`). The seed summary's `merged_by` says which ran.
3. A line's quantity is the most of its objects seen from any single panorama.

The model only groups ids. A line's wording is always one of the original detections and the
quantity is always computed by rule, so the merge can combine lines but cannot invent an item or
raise a count. Answers are cached per case.

## What the CoStar APIs actually do

Checked with real calls on Nov 14, 2026. These differ from the track's quickstarts:

- **Panorama to room**: `panoLocations { anchor { room { id } } }` links them directly. Panoramas
  with no room were taken outdoors. No polygon test is needed.
- **Rooms**: `rooms { tags }` includes `unfurnished` for empty rooms. `dimensionEstimates` is
  filled where the deprecated `dimensions` is missing; a height of 0 means "not measured". Room
  ids repeat between scans, so rooms are keyed by case.
- **Aerial dates**: `property/extract` returns one record per imagery flight (6 to 12 per LA
  parcel, 2020 to 2026) in no useful order, and `latest=true` does not narrow it. `features[0]` is
  not the newest. `ortho/extract` and `elements/detect` accept a `collection`, so Ashline uses
  the last flight before the loss date.
- **Yard detection**: cars and sheds are found. Grills and patio sets mostly are not at this
  resolution, and roof equipment gets mislabelled, so hits on the main roof are dropped. Pools,
  hot tubs, trampolines, playgrounds and decks are better read from the parcel's precomputed
  attributes, which the tool also returns.
- **Images**: scan faces are served from `cdn-2.matterport.com` through signed, expiring links.
  They are downloaded at once and the links are never stored.
- **CoStar kit**: it holds both formats. `residential-listings.csv` is a small index (scan id, listing ids,
  address); the records are JSONL under `data/`, with field docs in `models/` and `schemas/`.
  `ashline kit-info` lists it; `ashline listing --sid <scan id> --shape` shows a record's fields without values.

## Data rules

CoStar, Matterport and Vexcel data is for hackathon use only and must not be republished. This
repo is public, so:

- `cases/`, `.cache/`, any copy of `models_index.csv` and the kit are git-ignored. The scan index
  is fetched at runtime.
- No real address, scan id, floor plan or scan image belongs in code, tests, the README or the
  Devpost page. Tests use made-up data only. The demo home's scan id or address lives in `.env`.
- State on stage that the demo home is a stand-in.

Exports are estimates for planning, not legal advice; confirm with your adjuster.
