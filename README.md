# Ashline

Ashline is the family group chat that remembers your home. After a wildfire, relatives text old
photos, Ashline circles and prices what it sees, calls to help recall what no photo shows, and
builds the evidence-linked contents list an insurer asks for.

This repo currently holds the **worker** side of the **home twin seed**: given the survivor's
address, it pulls the Matterport 3D scan (inside) and the Vexcel aerial (outside) of that home and
turns both into the first batch of evidence for the case.

Everything the seed finds is a **suggestion**. A listing scan can show staging or a previous
owner's furniture, so nothing becomes an inventory item until the family confirms it.

## Setup

```bash
cd worker
python -m venv .venv && source .venv/bin/activate
pip install -e '.[agent,dev]'            # add ,local-vision to run item detection without an API key
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
```

`seed` is safe to run again: scan photos, detections and aerial results are cached per case, and
answers the family already gave are kept. Run it once before the demo to precompute the demo home.

## Tools

| Tool | Module | What it does |
| --- | --- | --- |
| `home_lookup` | `ashline/costar/index.py` | Address text or lat/lon to scan id, lat/lon, listing id and capture date. Returns "no scan" cleanly, and asks for the unit in multi-unit buildings. |
| `scan_rooms` | `ashline/costar/matterport.py` | Room list with dimensions, furnished flag and the photographer's capture notes. |
| `scan_photos` | `ashline/costar/matterport.py` | Picks about 10 of a scan's ~45 panoramas (big, valuable rooms first) and saves their four horizon faces, each tagged with scan, panorama, face and room. |
| `aerial_parcel` | `ashline/costar/vexcel.py` | Parcel, buildings, yard findings and an annotated aerial image from the last flight **before the loss date**. |
| `listing_record` | `ashline/costar/listing.py` | Optional CoStar listing join. Not verified against the real kit. |

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

`ASHLINE_DETECTOR=vlm` sends each photo to a vision model through any OpenAI-compatible endpoint
(`ASHLINE_VLM_BASE_URL`, `ASHLINE_VLM_API_KEY`, `ASHLINE_VLM_MODEL`). `ASHLINE_DETECTOR=florence`
runs Florence-2 locally: no key, about 15 seconds per photo on a laptop CPU, fewer small items
found. With no setting, the vision model is used when a key is set and the local model otherwise.

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
- **Not checked**: the CoStar kit (format and listing join), because no kit link was available.

## Data rules

CoStar, Matterport and Vexcel data is for hackathon use only and must not be republished. This
repo is public, so:

- `cases/`, `.cache/`, any copy of `models_index.csv` and the kit are git-ignored. The scan index
  is fetched at runtime.
- No real address, scan id, floor plan or scan image belongs in code, tests, the README or the
  Devpost page. Tests use made-up data only. The demo home's scan id or address lives in `.env`.
- State on stage that the demo home is a stand-in.

Exports are estimates for planning, not legal advice; confirm with your adjuster.
