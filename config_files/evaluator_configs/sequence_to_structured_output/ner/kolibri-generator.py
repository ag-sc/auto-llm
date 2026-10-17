from pathlib import Path

ROOT = Path(".")  # directory containing general/, history/, legal/, medical/, ...
OLD_NAME = "gemma-4-12B-it"
NEW_NAME = "kolibri-1"
OLD_MODEL = "google/gemma-4-12B-it"
NEW_MODEL = "Aleph-Alpha/Kolibri-1"

for src in sorted(ROOT.rglob(f"*{OLD_NAME}*.yaml")):
    dst = src.with_name(src.name.replace(OLD_NAME, NEW_NAME))
    text = src.read_text(encoding="utf-8")

    # The model id must be replaced first; the generic replacement comes second.
    text = text.replace(OLD_MODEL, NEW_MODEL)
    text = text.replace(OLD_NAME, NEW_NAME)

    dst.write_text(text, encoding="utf-8")
    print(f"{src} -> {dst}")
