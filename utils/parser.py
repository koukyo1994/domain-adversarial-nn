import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--skip-train", dest="skip", action="store_true")
    return parser
