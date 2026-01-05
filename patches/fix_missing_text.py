#!/usr/bin/env python3
"""
BabelDOC patch to fix missing text issue.

Problem: When DocLayout-YOLO fails to recognize certain text regions,
characters that fall outside detected layout areas are skipped entirely.

Solution:
1. Modify is_text_layout() to treat None layout as "plain text"
2. Handle None layout when accessing layout.name in create_paragraphs()

This is a known limitation of layout-based translation systems where
the layout detection model may not capture all text regions, especially:
- Text near page edges
- Text with unusual formatting (e.g., numbered items like "NO.1")
- Text that spans across detected region boundaries
"""

import sys
import re
from pathlib import Path


def patch_paragraph_finder():
    """Patch the paragraph_finder.py file to fix missing text issue."""

    # Find the babeldoc package location
    for path in sys.path:
        paragraph_finder_path = Path(path) / "babeldoc" / "document_il" / "midend" / "paragraph_finder.py"
        if paragraph_finder_path.exists():
            print(f"Found paragraph_finder.py at: {paragraph_finder_path}")
            break
    else:
        print("Error: Could not find babeldoc paragraph_finder.py")
        return False

    # Read the current content
    content = paragraph_finder_path.read_text(encoding='utf-8')

    # Check if already patched
    if "# PATCHED: Treat None layout as plain text" in content:
        print("File already patched, skipping.")
        return True

    patched = False

    # Patch 1: Fix is_text_layout method
    simple_old = 'return layout is not None and layout.name in ['
    simple_new = '''# PATCHED: Treat None layout as plain text
        if layout is None:
            return True
        return layout.name in ['''

    if simple_old in content:
        content = content.replace(simple_old, simple_new)
        print("Patched is_text_layout method")
        patched = True

    # Patch 2: Fix layout_label=current_layout.name when current_layout is None
    # There are multiple places where this pattern occurs
    old_layout_label = 'layout_label=current_layout.name,'
    new_layout_label = 'layout_label=current_layout.name if current_layout else "plain text",  # PATCHED'

    if old_layout_label in content:
        content = content.replace(old_layout_label, new_layout_label)
        print("Patched layout_label assignments")
        patched = True

    # Patch 3: Fix the conditional in create_paragraphs that uses char_layout.name and current_layout.name
    # Original pattern: layout_label=char_layout.name if not current_layout else current_layout.name,
    # Both char_layout and current_layout can be None, so we need to handle both cases
    old_cond = 'layout_label=char_layout.name\n                            if not current_layout\n                            else current_layout.name,'
    new_cond = 'layout_label=(char_layout.name if char_layout else "plain text")\n                            if not current_layout\n                            else (current_layout.name if current_layout else "plain text"),  # PATCHED'

    if old_cond in content:
        content = content.replace(old_cond, new_cond)
        print("Patched conditional layout name access (both branches)")
        patched = True

    # Patch 4: Fix the condition that compares char_layout.id with current_layout.id
    # When char_layout is None, accessing .id will fail
    old_id_check = 'or char_layout.id != current_layout.id'
    new_id_check = 'or (char_layout is not None and current_layout is not None and char_layout.id != current_layout.id)  # PATCHED'

    if old_id_check in content:
        content = content.replace(old_id_check, new_id_check)
        print("Patched char_layout.id comparison")
        patched = True

    # Patch 5: Fix the paragraph splitting logic when layout is None
    # Problem: When current_layout is None, EVERY character starts a new paragraph!
    # This causes each character to be in its own paragraph, which then gets skipped
    # because min_text_length=5 (text too short to translate).
    #
    # Original logic: current_layout is None -> always start new paragraph
    # Fixed logic: Only start new paragraph when:
    #   - This is the first character (no current_line_chars), OR
    #   - Layout state changes AND not on the same line, OR
    #   - Layout IDs are different AND not on the same line
    old_split_logic = '''if not (is_small_char and current_line_chars) and (
                current_layout is None
                or (char_layout is not None and current_layout is not None and char_layout.id != current_layout.id)  # PATCHED'''
    new_split_logic = '''if not (is_small_char and current_line_chars) and (
                (not current_line_chars)  # First character, need to initialize paragraph  # PATCHED
                or (  # Layout state changed, but only split if NOT on same line  # PATCHED
                    ((current_layout is None) != (char_layout is None))
                    and current_line_chars
                    and abs((char.box.y + char.box.y2) / 2 - (current_line_chars[-1].box.y + current_line_chars[-1].box.y2) / 2) >= max(char.box.y2 - char.box.y, current_line_chars[-1].box.y2 - current_line_chars[-1].box.y)  # PATCHED
                )
                or (  # Different layout IDs, but only split if NOT on same line  # PATCHED
                    char_layout is not None and current_layout is not None and char_layout.id != current_layout.id
                    and current_line_chars
                    and abs((char.box.y + char.box.y2) / 2 - (current_line_chars[-1].box.y + current_line_chars[-1].box.y2) / 2) >= max(char.box.y2 - char.box.y, current_line_chars[-1].box.y2 - current_line_chars[-1].box.y)  # PATCHED
                )'''

    if old_split_logic in content:
        content = content.replace(old_split_logic, new_split_logic)
        print("Patched paragraph splitting logic for None layout")
        patched = True
    else:
        # Try the original unpatched version
        old_split_logic_orig = '''if not (is_small_char and current_line_chars) and (
                current_layout is None
                or char_layout.id != current_layout.id'''
        new_split_logic_orig = '''if not (is_small_char and current_line_chars) and (
                (not current_line_chars)  # First character, need to initialize paragraph  # PATCHED
                or (  # Layout state changed, but only split if NOT on same line  # PATCHED
                    ((current_layout is None) != (char_layout is None))
                    and current_line_chars
                    and abs((char.box.y + char.box.y2) / 2 - (current_line_chars[-1].box.y + current_line_chars[-1].box.y2) / 2) >= max(char.box.y2 - char.box.y, current_line_chars[-1].box.y2 - current_line_chars[-1].box.y)  # PATCHED
                )
                or (  # Different layout IDs, but only split if NOT on same line  # PATCHED
                    char_layout is not None and current_layout is not None and char_layout.id != current_layout.id
                    and current_line_chars
                    and abs((char.box.y + char.box.y2) / 2 - (current_line_chars[-1].box.y + current_line_chars[-1].box.y2) / 2) >= max(char.box.y2 - char.box.y, current_line_chars[-1].box.y2 - current_line_chars[-1].box.y)  # PATCHED
                )'''

        if old_split_logic_orig in content:
            content = content.replace(old_split_logic_orig, new_split_logic_orig)
            print("Patched paragraph splitting logic for None layout (original version)")
            patched = True

    # Patch 6: Fix xobj_id splitting for characters on the same line
    # Problem: When "practices" spans different xobjects, it gets split into "prac" + "tices"
    # The "tices" paragraph is then not properly handled/translated.
    #
    # Solution: If characters are on the same line (similar y coordinate),
    # don't split the paragraph even if xobj_id is different.
    # This is more aggressive but necessary to keep words together.
    old_xobj_check = '''or (  # 不是同一个 xobject
                    current_line_chars
                    and current_line_chars[-1].xobj_id != char.xobj_id
                )'''
    new_xobj_check = '''or (  # 不是同一个 xobject，但同行字符例外  # PATCHED
                    current_line_chars
                    and current_line_chars[-1].xobj_id != char.xobj_id
                    and not (  # 如果在同一行，則不分割（無論 layout 狀態）  # PATCHED
                        abs((char.box.y + char.box.y2) / 2 - (current_line_chars[-1].box.y + current_line_chars[-1].box.y2) / 2) < max(
                            char.box.y2 - char.box.y,
                            current_line_chars[-1].box.y2 - current_line_chars[-1].box.y
                        )  # PATCHED: same line (y center within character height)
                    )
                )'''

    if old_xobj_check in content:
        content = content.replace(old_xobj_check, new_xobj_check)
        print("Patched xobj_id splitting for adjacent characters")
        patched = True

    if not patched:
        print("Warning: No patterns found to patch")
        return False

    # Write the patched content
    paragraph_finder_path.write_text(content, encoding='utf-8')
    print("Successfully patched paragraph_finder.py")
    return True


if __name__ == "__main__":
    success = patch_paragraph_finder()
    sys.exit(0 if success else 1)
