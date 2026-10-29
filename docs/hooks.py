"""MkDocs hooks for the diff-soup docs."""

import posixpath
import re

_HREF = re.compile(r'href="([^":#]+\.(?:md|ipynb))(#[^"]*)?"')


def on_post_page(output, page, config):
    """Rewrite relative ``.md`` links inside rendered notebooks to site URLs.

    mkdocs-jupyter renders a notebook's markdown cells without MkDocs' link
    resolution, so ``../overview/x.md`` would point at a missing file.
    """
    if not page.file.src_uri.endswith(".ipynb"):
        return output
    files = _FILES.get("files")

    def fix(match):
        """Return the matched link pointing at the rendered page, or unchanged."""
        target = posixpath.normpath(
            posixpath.join(posixpath.dirname(page.file.src_uri), match.group(1))
        )
        file = files.get_file_from_path(target) if files else None
        if file is None:
            return match.group(0)
        return f'href="{file.url_relative_to(page.file)}{match.group(2) or ""}"'

    return _HREF.sub(fix, output)


_FILES = {}


def on_files(files, config):
    """Keep the site's file collection for `on_post_page`."""
    _FILES["files"] = files
    return files
