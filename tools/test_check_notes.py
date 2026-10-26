#!/usr/bin/env python3
"""Guard the checker scope after adding website dependencies and builds."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from check_notes import markdown_files, check_math


class CheckerScope(unittest.TestCase):
    def test_maintained_documents_included_outputs_excluded(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            kept = ['README.md', 'notes/Lecture-01.md', 'website/README.md', 'website/docs/review.md']
            ignored = ['website/node_modules/pkg/README.md', 'website/dist/resources/notes/copy.md',
                       'website/test-results/error-context.md', '.git/cache.md']
            for name in kept + ignored:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# test\n')
            self.assertEqual({str(p.relative_to(root)) for p in markdown_files(root)}, set(kept))

    def test_math_errors_still_detected(self):
        errors = []
        check_math(Path('note.md'), 'unclosed $x\n', errors)
        self.assertTrue(errors)


if __name__ == '__main__':
    unittest.main()
