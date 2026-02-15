from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml_core.registry.versioning import promote_to_prod, read_aliases, rollback_prod, set_alias


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage local model registry aliases.")
    parser.add_argument("--registry-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)

    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser("show")
    show_parser.set_defaults(action="show")

    set_parser = subparsers.add_parser("set")
    set_parser.add_argument("--alias", choices=["staging", "prod"], required=True)
    set_parser.add_argument("--version", required=True)
    set_parser.set_defaults(action="set")

    promote_parser = subparsers.add_parser("promote-prod")
    promote_parser.add_argument("--version", required=True)
    promote_parser.set_defaults(action="promote")

    rollback_parser = subparsers.add_parser("rollback-prod")
    rollback_parser.set_defaults(action="rollback")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    registry_dir = Path(args.registry_dir)
    model_name = args.model_name

    if args.action == "show":
        aliases = read_aliases(registry_dir=registry_dir, model_name=model_name)
    elif args.action == "set":
        aliases = set_alias(
            registry_dir=registry_dir,
            model_name=model_name,
            alias=args.alias,
            version=args.version,
        )
    elif args.action == "promote":
        aliases = promote_to_prod(
            registry_dir=registry_dir,
            model_name=model_name,
            version=args.version,
        )
    elif args.action == "rollback":
        aliases = rollback_prod(
            registry_dir=registry_dir,
            model_name=model_name,
        )
    else:
        raise RuntimeError("Unknown command")

    print(json.dumps(aliases, indent=2))


if __name__ == "__main__":
    main()
