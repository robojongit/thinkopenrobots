"""Static checks on the site's pages and scripts."""

import re
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parent.parent
SCRIPTS = ["app.js", "motion.js"]
PAGES = ["index.html"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_scripts_never_parse_html(name):
    """leaderboard.json is untrusted: text goes through textContent only."""
    src = (SITE / name).read_text()
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in src, f"{name} uses {sink}"


@pytest.mark.parametrize("name", PAGES)
def test_pages_load_only_local_versioned_assets(name):
    html = (SITE / name).read_text()
    refs = re.findall(r'<(?:script|link)[^>]+(?:src|href)="([^"]+)"', html)
    assert refs
    for ref in refs:
        assert not re.match(r"(https?:)?//", ref), f"{name} loads {ref} from another origin"
        if ref.endswith((".js", ".css")) or ".js?" in ref or ".css?" in ref:
            assert "?v=" in ref, f"{name}: {ref} has no cache-busting version"


def test_fonts_ship_with_their_licences():
    fonts = sorted((SITE / "fonts").glob("*.woff2"))
    assert len(fonts) == 2
    licences = [p.read_text() for p in (SITE / "fonts").glob("OFL-*.txt")]
    assert len(licences) == 2 and all("SIL Open Font License" in t for t in licences)
