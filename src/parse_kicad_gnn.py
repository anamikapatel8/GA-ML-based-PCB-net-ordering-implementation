# parse_kicad.py
import re, json, os

def extract_blocks(text, start_token="(footprint", end_token=")"):
    """Extract balanced blocks like '(footprint ... )' or '(module ... )' safely."""
    blocks = []
    stack = 0
    start = None
    L = len(text)
    i = 0
    while i < L:
        if text[i:i+len(start_token)] == start_token:
            if stack == 0:
                start = i
            stack += 1
            i += len(start_token)
            continue
        ch = text[i]
        if ch == '(':
            if stack > 0:
                stack += 1
        elif ch == ')':
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    blocks.append(text[start:i+1])
                    start = None
        i += 1
    return blocks


def parse_kicad(filepath):
    """
    Parse a KiCad PCB file for:
      - board_bbox (xmin, ymin, xmax, ymax)
      - nets : mapping net_name -> list of (x,y) pad coordinates (legacy compatibility)
      - net_pads : mapping net_name -> list of pad dicts (with component id)
      - components : mapping comp_ref -> { 'x': fx, 'y': fy, 'pads': [...], 'nets': set(...) }
      - pad_count : integer
    This version preserves previous outputs (so other code using 'nets' will still work),
    and adds richer structures useful for graph building (shared-component edges, edge features).
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Detect legacy KiCad 4 format (uses "module")
    old_format = "(module " in content

    # --- Board outline (Edge.Cuts or legacy area) ---
    board_bbox = None

    # --- Board outline (Edge.Cuts) via rectangle quick match ---
    rect = re.search(
        r'\(gr_rect\s+\(start\s+([-\d\.]+)\s+([-\d\.]+)\)\s+\(end\s+([-\d\.]+)\s+([-\d\.]+)\)',
        content
    )
    if rect:
        board_bbox = [float(rect.group(i)) for i in range(1, 5)]
    else:
        xs, ys = [], []
        for m in re.finditer(
            r'\(gr_(?:line|arc|circle)[\s\S]*?\(layer\s+"?Edge\.Cuts"?\)[\s\S]*?\(start\s+([-\d\.]+)\s+([-\d\.]+)\)[\s\S]*?\(end\s+([-\d\.]+)\s+([-\d\.]+)\)',
            content
        ):
            xs += [float(m.group(1)), float(m.group(3))]
            ys += [float(m.group(2)), float(m.group(4))]

        for m in re.finditer(
            r'\(gr_circle[\s\S]*?\(layer\s+"?Edge\.Cuts"?\)[\s\S]*?\(center\s+([-\d\.]+)\s+([-\d\.]+)\)\s+\(end\s+([-\d\.]+)\s+([-\d\.]+)\)',
            content
        ):
            cx, cy, ex, ey = map(float, m.groups())
            r = ((ex - cx)**2 + (ey - cy)**2)**0.5
            xs += [cx - r, cx + r]
            ys += [cy - r, cy + r]

        if xs and ys:
            board_bbox = [min(xs), min(ys), max(xs), max(ys)]

    # Fallback for legacy KiCad v4 "(area ...)" field
    if board_bbox is None:
        area_match = re.search(r'\(area\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\)', content)
        if area_match:
            board_bbox = [float(area_match.group(i)) for i in range(1, 5)]

    # --- Nets (supports numeric id + quoted/unquoted names) ---
    nets = {}
    for m in re.finditer(r'\(net\s+(\d+)\s+"?([^"\)]+)"?\)', content):
        nets[m.group(1)] = m.group(2)

    # --- Footprints / modules blocks ---
    footprints = extract_blocks(content, "(module" if old_format else "(footprint", ")")

    pads = []
    # Pattern extracts: pad name, (at x y), net id, net name
    pad_pattern = re.compile(
        r'\(pad\s+("?[^"\s]+?"?)[\s\S]*?\(at\s+([-\d\.]+)\s+([-\d\.]+)(?:\s+[^\)]*)?\)[\s\S]*?\(net\s+(\d+)\s+"?([^"\)]+)"?\)',
        re.MULTILINE
    )

    components = {}  # comp_ref -> {x,y, pads: [pad dict], nets: set()}
    comp_counter = 0

    for idx, fp in enumerate(footprints):
        # try to locate component origin (fx, fy)
        at_match = re.search(r'\(at\s+([-\d\.]+)\s+([-\d\.]+)', fp)
        fx, fy = (float(at_match.group(1)), float(at_match.group(2))) if at_match else (0.0, 0.0)

        # try to find a human-friendly reference name for the component
        # multiple possible formats across KiCad versions:
        ref_match = re.search(r'\(fp_text\s+reference\s+"?([^"\)]+)"?\)', fp)
        if not ref_match:
            ref_match = re.search(r'\(reference\s+"?([^"\)]+)"?\)', fp)
        if not ref_match:
            # as a fallback try to capture the module/footprint tag name
            header_match = re.match(r'\(\s*(?:module|footprint)\s+([^\s\)]+)', fp)
            comp_ref = header_match.group(1) + f"_{idx}" if header_match else f"COMP_{comp_counter}"
        else:
            comp_ref = ref_match.group(1).strip()
        comp_counter += 1

        # prepare component entry
        components[comp_ref] = {
            "x": fx,
            "y": fy,
            "pads": [],
            "nets": set()
        }

        # find pads in this footprint block
        for m in pad_pattern.finditer(fp):
            pad_name = m.group(1).strip('"')
            local_x, local_y = float(m.group(2)), float(m.group(3))
            world_x = fx + local_x
            world_y = fy + local_y
            net_id = m.group(4)
            net_name = m.group(5).strip('"')

            pad = {
                "pad": pad_name,
                "x": world_x,
                "y": world_y,
                "net_id": net_id,
                "net_name": net_name,
                "comp": comp_ref
            }
            pads.append(pad)
            components[comp_ref]["pads"].append(pad)
            components[comp_ref]["nets"].add(net_name)

    # --- Group pads per net (detailed list) ---
    net_pads = {}
    for p in pads:
        net_pads.setdefault(p["net_name"], []).append(p)

    # --- Legacy mapping: nets -> list of (x,y) coordinates (keeps compatibility) ---
    nets_dict = {}
    for net_name, plist in net_pads.items():
        coords = [(float(p["x"]), float(p["y"])) for p in plist]
        nets_dict[net_name] = coords

    result = {
        "board_bbox": board_bbox,
        "nets": nets_dict,          # legacy/simple mapping: net_name -> [(x,y), ...]
        "net_pads": net_pads,      # detailed mapping: net_name -> [ {pad, x,y, net_id, comp}, ... ]
        "components": components,  # comp_ref -> {x,y, pads: [...], nets: set(...) }
        "pad_count": len(pads)
    }
    return result


if __name__ == "__main__":
    pcb_path = "../data/d1_mini_shield.kicad_pcb"
    result = parse_kicad(pcb_path)
    os.makedirs("../results", exist_ok=True)
    with open("../results/board_meta.json", "w") as f:
        # convert sets to lists for JSON compatibility
        res_copy = result.copy()
        res_copy["components"] = {
            k: {**v, "nets": list(v["nets"])} for k, v in result["components"].items()
        }
        json.dump(res_copy, f, indent=2)
    print("Parsed:", pcb_path)
    print(f"Pads: {result['pad_count']}, Nets: {len(result['nets'])}")
    print("Net names (sample):", list(result["nets"].keys())[:10])
