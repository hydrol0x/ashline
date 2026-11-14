"""Command line for the home twin seed.

    ashline lookup "<address>"
    ashline seed --case demo                 # home from ASHLINE_DEMO_SID or ASHLINE_DEMO_ADDRESS in .env
    ashline seed --case demo --sid <scan id>
    ashline seed --case fam1 --address "..." --loss-date 2025-01-07
    ashline questions --case demo
    ashline export --case demo --include-suggested
    ashline kit-info
"""
from __future__ import annotations

import argparse
import json

from . import config
from .costar.index import home_lookup
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

    p = sub.add_parser("questions", help="voice-call questions from gaps in the case")
    p.add_argument("--case", required=True)

    p = sub.add_parser("export", help="write the inventory CSV")
    p.add_argument("--case", required=True)
    p.add_argument("--include-suggested", action="store_true")

    sub.add_parser("kit-info", help="list what is inside the CoStar kit (settles its format)")
    p = sub.add_parser("listing", help="CoStar listing for a homes_listing_id")
    p.add_argument("listing_id")

    p = sub.add_parser("delete", help="delete every stored row of a case")
    p.add_argument("--case", required=True)

    args = parser.parse_args(argv)
    try:
        _run(args, parser)
    except config.MissingKey as exc:
        parser.exit(2, f"{exc}\nLooked for .env in: {', '.join(str(p) for p in config._dotenv_paths())}\n")


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
        result = seed_home_twin(args.case, address, args.lat, args.lon, args.loss_date,
                                run_detection=not args.no_detect, max_faces=args.max_faces, sid=sid)
        print(json.dumps(result, indent=1))
    elif args.cmd == "questions":
        for q in gap_questions(args.case):
            print(f"[{q['kind']:<9} {q['weight']:>6}] {q['text']}")
    elif args.cmd == "export":
        print(export_csv(args.case, include_suggested=args.include_suggested))
    elif args.cmd == "kit-info":
        from .costar.listing import kit_info
        print(json.dumps(kit_info(), indent=1))
    elif args.cmd == "listing":
        from .costar.listing import listing_record
        print(json.dumps(listing_record(args.listing_id), indent=1))
    elif args.cmd == "delete":
        Store().delete_case(args.case)
        print(f"case {args.case} deleted")


if __name__ == "__main__":
    main()
