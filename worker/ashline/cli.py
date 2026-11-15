"""Command line for the home twin seed.

    ashline lookup "<address>"
    ashline seed --case demo                 # home from ASHLINE_DEMO_SID or ASHLINE_DEMO_ADDRESS in .env
    ashline seed --case demo --sid <scan id>
    ashline seed --case fam1 --address "..." --loss-date 2025-01-07
    ashline questions --case demo
    ashline export --case demo --include-suggested
    ashline kit-info
    ashline chat --as +15550001 --name Mom      # talk to Ashline in the terminal
    ashline serve                               # the inventory page, on http://127.0.0.1:8000
    ashline page --case demo                    # the inventory page links
"""
from __future__ import annotations

import argparse
import json
import shutil

from . import config
from .costar.index import home_lookup
from .detect import DetectionError
from .export import export_csv
from .gaps import gap_questions
from .seed import seed_home_twin
from .store import Store


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ashline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("lookup", help="find the 3D scan for an address")
    p.add_argument("address", nargs="?")
    p.add_argument("--sid", help="look a scan up by its id instead")
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)

    p = sub.add_parser("seed", help="run the home twin seed for a case")
    p.add_argument("--case", required=True)
    p.add_argument("--address")
    p.add_argument("--sid", help="scan id from models_index.csv, instead of an address")
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--loss-date")
    p.add_argument("--max-faces", type=int, default=40)
    p.add_argument("--no-detect", action="store_true", help="download photos but skip item detection")
    p.add_argument("--redetect", action="store_true", help="discard cached detections and run the detector again")

    p = sub.add_parser("questions", help="voice-call questions from gaps in the case")
    p.add_argument("--case", required=True)

    p = sub.add_parser("export", help="write the inventory CSV")
    p.add_argument("--case", required=True)
    p.add_argument("--include-suggested", action="store_true")

    sub.add_parser("kit-info", help="list what is inside the CoStar kit (settles its format)")
    p = sub.add_parser("listing", help="CoStar listing behind a scan")
    p.add_argument("listing_id", nargs="?", help="homes_listing_id from the scan index")
    p.add_argument("--sid", help="scan id")
    p.add_argument("--shape", action="store_true", help="print only field names and types, no values")
    p.add_argument("--summary", action="store_true", help="print the short summary Ashline uses")

    p = sub.add_parser("chat", help="talk to Ashline in the terminal, as any family member")
    p.add_argument("--as", dest="phone", default="+15550001", help="the phone number you are texting from")
    p.add_argument("--name", default=None)
    p.add_argument("--url", default=None, help="public address of the inventory page, for links")

    p = sub.add_parser("serve", help="run the inventory page and the bridge endpoints")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept connections from a tunnel or phone")

    p = sub.add_parser("price-check", help="refresh the estimates, then look up the most valuable items' prices online")
    p.add_argument("--case", required=True)
    p.add_argument("--top", type=int, default=20, help="how many items to look up, most valuable first")
    p.add_argument("-v", "--verbose", action="store_true", help="show each step: what was searched, kept and dropped")

    sub.add_parser("photon", help="run the conversation on Photon 2.0 (iMessage)")
    sub.add_parser("telegram", help="run the conversation on Telegram (works in family group chats)")

    p = sub.add_parser("bank", help="match the family's card purchases (Capital One Nessie) to inventory items")
    p.add_argument("--case", required=True)
    p.add_argument("--apply", action="store_true", help="use the matched purchases as the items' prices")
    p.add_argument("--seed-demo", type=int, metavar="N",
                   help="demo only: first write purchases for the N most valuable items into the mock bank")

    p = sub.add_parser("orders", help="match the lines of an Amazon order-history CSV to inventory items")
    p.add_argument("--case", required=True)
    p.add_argument("file", help="the order-history file, e.g. Retail.OrderHistory.1.csv")
    p.add_argument("--apply", action="store_true", help="use the matched order lines as the items' prices")

    p = sub.add_parser("plan", help="draw the 2D plan of a case: from its scan, or sketched from its photos")
    p.add_argument("--case", required=True)
    p.add_argument("--png", help="also save the plan as an image at this path")
    p.add_argument("--redraw", action="store_true", help="ask the model for a new sketch even if nothing changed")

    p = sub.add_parser("page", help="print the inventory page links of a case")
    p.add_argument("--case", required=True)

    p = sub.add_parser("delete", help="delete every stored row of a case")
    p.add_argument("--case", required=True)

    args = parser.parse_args(argv)
    try:
        _run(args, parser)
    except config.MissingKey as exc:
        parser.exit(2, f"{exc}\nLooked for .env in: {', '.join(str(p) for p in config._dotenv_paths())}\n")
    except DetectionError as exc:
        parser.exit(1, f"Item detection stopped: {exc}\nPhotos already done are cached; run the command again to continue.\n")


def _run(args, parser) -> None:
    if args.cmd == "lookup":
        print(json.dumps(home_lookup(args.address, args.lat, args.lon, sid=args.sid), indent=1))
    elif args.cmd == "seed":
        address, sid = args.address, args.sid
        if not (address or sid or args.lat is not None):      # nothing given: fall back to the demo home in .env
            sid, address = config.env("ASHLINE_DEMO_SID"), config.env("ASHLINE_DEMO_ADDRESS")
        if not (address or sid or args.lat is not None):
            parser.exit(2, "No home given. Pass --address, --sid or --lat/--lon, or set ASHLINE_DEMO_SID "
                           "or ASHLINE_DEMO_ADDRESS in .env.\n"
                           f"Looked for .env in: {', '.join(str(p) for p in config._dotenv_paths())}\n")
        if args.redetect:
            shutil.rmtree(config.case_dir(args.case) / "detections", ignore_errors=True)
        result = seed_home_twin(args.case, address, args.lat, args.lon, args.loss_date,
                                run_detection=not args.no_detect, max_faces=args.max_faces, sid=sid)
        if not args.no_detect:
            from .pricing import price_items
            result["pricing"] = price_items(args.case)
        print(json.dumps(result, indent=1))
    elif args.cmd == "questions":
        _known(args.case, parser)
        for q in gap_questions(args.case):
            print(f"[{q['kind']:<9} {q['weight']:>6}] {q['text']}")
    elif args.cmd == "export":
        _known(args.case, parser)
        print(export_csv(args.case, include_suggested=args.include_suggested))
    elif args.cmd == "kit-info":
        from .costar.listing import kit_info
        print(json.dumps(kit_info(), indent=1))
    elif args.cmd == "listing":
        from .costar.listing import listing_record, shape
        record = listing_record(args.listing_id, sid=args.sid)
        if args.shape and record.get("found"):
            print("\n".join(shape({"listing": record["listing"], "features": record["features"]})))
        elif args.summary and record.get("found"):
            from .costar.listing import summarize
            print(json.dumps(summarize(record), indent=1))
        else:
            print(json.dumps(record, indent=1))
    elif args.cmd == "chat":
        simulate(args.phone, args.name, args.url)
    elif args.cmd == "serve":
        try:
            import uvicorn

            from .web import app
        except ImportError as exc:
            parser.exit(1, f"The page needs packages that are not installed ({exc.name}). From worker/, run:\n"
                           "  pip install -e '.[agent,claude,dev]'\n")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    elif args.cmd == "price-check":
        from .price_search import price_check
        _known(args.case, parser)
        from .pricing import price_items as refresh_estimates
        refreshed = refresh_estimates(args.case)           # rough figures are worked out again first
        print(f"estimates refreshed: {refreshed['priced']} items ({refreshed['source'] or 'nothing to do'})")
        from .price_search import get_searcher
        searcher = get_searcher()
        if searcher is not None and args.verbose:
            searcher.trace = print                         # what was looked for, found, kept and dropped
        if searcher is not None and not args.verbose:
            print(f"Looking up to {args.top} items up online, three at a time. Each takes a minute or two; "
                  "a line appears as each one finishes.", flush=True)
        try:
            print(json.dumps(price_check(args.case, top=args.top, searcher=searcher, report=print), indent=1))
        except KeyboardInterrupt:
            parser.exit(130, "\nStopped. Prices recorded so far are kept.\n")
    elif args.cmd == "photon":
        from .bridge_photon import PhotonBridge
        PhotonBridge().run()
    elif args.cmd == "telegram":
        from .bridge_telegram import TelegramBridge
        TelegramBridge().run()
    elif args.cmd == "bank":
        from . import bank
        store = Store()
        if not store.get_case(args.case):
            parser.exit(1, f"No case {args.case!r}.\n")
        try:
            client = bank.NessieClient()
            if args.seed_demo:
                made = bank.seed_demo(args.case, store, client, top=args.seed_demo)
                print(f"Demo purchases written to the mock bank: {len(made['purchases'])}")
                for row in made["purchases"]:
                    print(f"  {row['amount']:>9.2f}  {row['merchant']:<12} {row['item']}")
                print(f"Add to .env so only this account is read:\n  NESSIE_CUSTOMER_ID={made['customer_id']}\n"
                      f"  NESSIE_ACCOUNT_ID={made['account_id']}")
                matches = bank.match_purchases(
                    [i for i in store.get_items(args.case, canonical_only=True) if i["status"] != "rejected"],
                    bank.purchases(client, account_id=made["account_id"]))
            else:
                matches = bank.find_matches(args.case, store, client)
        except (config.MissingKey, bank.BankError) as exc:
            parser.exit(1, f"{exc}\n")
        print(f"{len(matches)} purchase(s) match items on the list" + (":" if matches else "."))
        for match in matches:
            print(f"  {bank.describe(match)}")
        if args.apply and matches:
            print(f"Prices updated on {bank.apply_matches(store, args.case, matches)} item(s).")
        elif matches:
            print("Run again with --apply to use these as the prices, or text 'check my bank' to Ashline.")
    elif args.cmd == "orders":
        from . import bank, orders
        store = Store()
        if not store.get_case(args.case):
            parser.exit(1, f"No case {args.case!r}.\n")
        try:
            data = open(args.file, "rb").read()
        except OSError as exc:
            parser.exit(1, f"Could not read {args.file}: {exc.strerror}\n")
        if not orders.looks_like_orders(args.file, data):
            parser.exit(1, f"{args.file} does not look like an order history: no order id and product name columns.\n")
        lines = orders.read_orders(data)
        matches = orders.find_matches(args.case, store, lines)
        print(f"Read {len(lines)} paid order line(s). {len(matches)} match items on the list" + (":" if matches else "."))
        for match in matches:
            print(f"  {bank.describe(match)}")
        if args.apply and matches:
            print(f"Prices updated on {bank.apply_matches(store, args.case, matches)} item(s).")
        elif matches:
            print("Run again with --apply to use these as the prices.")
    elif args.cmd == "plan":
        from . import layout, sketch_claude
        store = Store()
        case = store.get_case(args.case)
        if not case:
            parser.exit(1, f"No case {args.case!r}.\n")
        if sketch_claude.available():
            try:
                sketch_claude.refresh(args.case, store, force=args.redraw)
            except sketch_claude.SketchError as exc:
                print(f"{exc} Using the simple sketch.")
        drawn = layout.build(args.case, store)
        print(f"{drawn['kind']} plan: {len(drawn['rooms'])} rooms, {len(drawn['pins']) - drawn['unplaced']} items placed"
              + (f", {drawn['unplaced']} with no known room" if drawn["unplaced"] else ""))
        if args.png:
            try:
                print(f"saved {layout.render_png(drawn, args.png)}")
            except (OSError, ValueError) as exc:           # a folder that is not there, or a name that is not a picture
                parser.exit(1, f"Cannot write {args.png}: {exc}\n")
        if case.get("owner_token"):
            base = (config.env("ASHLINE_PUBLIC_URL") or "http://127.0.0.1:8000").rstrip("/")
            print(f"page: {base}/p/{case['owner_token']}/plan")
    elif args.cmd == "page":
        store = Store()
        case = store.get_case(args.case)
        if not case:
            parser.exit(1, f"No case {args.case!r}.\n")
        if not case.get("owner_token"):
            import secrets
            store.set_case(args.case, owner_token=secrets.token_urlsafe(24), member_token=secrets.token_urlsafe(24),
                           name=case.get("name") or "Your home")
            case = store.get_case(args.case)
        base = (config.env("ASHLINE_PUBLIC_URL") or "http://127.0.0.1:8000").rstrip("/")
        print(f"owner:  {base}/p/{case['owner_token']}\nfamily: {base}/p/{case['member_token']}")
    elif args.cmd == "delete":
        _known(args.case, parser)
        Store().delete_case(args.case)
        print(f"case {args.case} deleted")


def _known(case_id: str, parser) -> dict:
    """The case, or a plain message and exit: a mistyped id must not quietly make an empty case."""
    case = Store().get_case(case_id)
    if not case:
        parser.exit(1, f"No case {case_id!r}.\n")
    return case


SIM_HELP = """Type a message and press enter. Lines starting with / are simulator commands:
  /photo a.jpg b.jpg     send photos
  /video walk.mp4        send a video (needs ffmpeg)
  /receipt r.jpg         send a receipt, order page or PDF invoice
  /wait [seconds]        let time pass (default: just past the photo quiet window) and run what is due
  /as +15550002 Maya     become another family member
  /group rivera          talk in a group chat called "rivera"   (/group off: back to 1:1)
  /quit"""


def simulate(phone: str, name: str | None, url: str | None, lines=None, out=print) -> None:
    """A terminal stand-in for the messaging bridge: the same engine, no Photon needed."""
    import sys
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    from .chat import Engine, Inbound

    engine = Engine(public_url=url or config.env("ASHLINE_PUBLIC_URL") or "http://127.0.0.1:8000")
    offset, group = timedelta(0), None
    interactive = lines is None and sys.stdin.isatty()
    if interactive:
        out(SIM_HELP)

    def now():
        return datetime.now(timezone.utc) + offset

    def show(messages):
        for m in messages:
            out(f"Ashline → {m.chat_id}: {m.text}")
            for path in ([m.image] if m.image else []) + list(m.files):
                out(f"          [attached: {path}]")
            if m.choices:
                out("          " + " ".join(f"[{label}]" for label, _ in m.choices))

    for line in (lines if lines is not None else sys.stdin):
        line = line.strip()
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line.startswith("/as "):
            parts = line.split(maxsplit=2)
            phone, name = parts[1], parts[2] if len(parts) > 2 else None
            out(f"(you are now {name or phone})")
        elif line.startswith("/group"):
            group = None if line.split()[-1] in ("off", "/group") else line.split(maxsplit=1)[1]
            out(f"(talking in {'the group ' + group if group else 'a 1:1 chat'})")
        elif line.startswith("/wait"):
            parts = line.split()
            offset += timedelta(seconds=float(parts[1]) if len(parts) > 1 else engine.batch_seconds + 1)
            show(engine.tick(now()))
        elif line.startswith(("/photo", "/video", "/receipt")):
            files = []
            for path in line.split()[1:]:
                try:
                    files.append((Path(path).name, Path(path).expanduser().read_bytes()))
                except OSError as exc:
                    out(f"(could not read {path}: {exc})")
            if files:
                caption = "receipt" if line.startswith("/receipt") else ""
                show(engine.handle(Inbound(group or phone, phone, caption, files, bool(group), name, now())))
        elif line.startswith("/"):
            out(SIM_HELP)
        else:
            show(engine.handle(Inbound(group or phone, phone, line, [], bool(group), name, now())))


if __name__ == "__main__":
    main()
