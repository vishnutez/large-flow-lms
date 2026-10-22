"""Embed decoding traces into the trajectory dashboard.

Writes two files: the page body for publishing as an Artifact, and a
standalone HTML file that opens directly in a browser.

    python scripts/build_dashboard.py output/traces/lafa-8B-*-q{0,2,4}.json \
        --output output/dashboard/lafa-trajectories.html
"""
import argparse
import json
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / 'dashboard' / 'template.html'
MODE_ORDER = {'flow': 0, 'speculative': 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('traces', nargs='+', help='Trace JSON files from trace_generation.py')
    parser.add_argument('--output', default='output/dashboard/lafa-trajectories.html')
    args = parser.parse_args()

    traces = [json.loads(Path(p).read_text()) for p in args.traces]
    traces.sort(key=lambda t: (t['meta']['qid'], MODE_ORDER.get(t['meta']['mode'], 9)))
    # Keep the embedded JSON from closing its own <script> element.
    payload = json.dumps(traces, separators=(',', ':')).replace('</', '<\\/')
    page = TEMPLATE.read_text().replace('/*__TRACES__*/', payload)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    standalone = out.with_name(out.stem + '.standalone.html')
    standalone.write_text('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
                          '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
                          '</head><body>\n' + page + '\n</body></html>\n')
    for path in (out, standalone):
        print(f'{path}: {path.stat().st_size / 1e6:.2f} MB, {len(traces)} traces')


if __name__ == '__main__':
    main()
