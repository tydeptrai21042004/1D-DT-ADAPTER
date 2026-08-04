#!/usr/bin/env python3
"""Apply this changed-files package to the original uploaded repository."""
from pathlib import Path
import json, shutil, sys
package=Path(__file__).resolve().parent
target=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path.cwd().resolve()
manifest=json.loads((package/"CHANGE_MANIFEST.json").read_text())
for rel in manifest["deleted"]:
 p=target/rel
 if p.is_file() or p.is_symlink(): p.unlink()
 elif p.is_dir(): shutil.rmtree(p)
for rel in manifest["added"]+manifest["modified"]:
 src=package/rel;dst=target/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
print(f"Applied {len(manifest['added'])} additions, {len(manifest['modified'])} modifications, and {len(manifest['deleted'])} deletions to {target}")
