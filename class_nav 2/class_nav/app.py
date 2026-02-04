import matplotlib
matplotlib.use("Agg")
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import io
import os
import difflib
import math

app = Flask(__name__)
CORS(app)

# --- CONFIG ---
ROOM_TYPE = "Type"
ROOM_NAME = "Room no."   # label column from GeoJSON
GEOJSON_PATHS = {
    "Level_1": "geojsons/ground_floor.geojson",
    "Level_2": "geojsons/first_floor.geojson",
    "Level_3": "geojsons/second_floor.geojson",
}

# --- SAFE LOAD GEOJSONS ---
floor_gdfs = {}
for lvl, path in GEOJSON_PATHS.items():
    if not os.path.exists(path):
        print(f"⚠️ GeoJSON missing: {path} (level {lvl}) - creating empty GeoDataFrame")
        floor_gdfs[lvl] = gpd.GeoDataFrame()
        continue
    try:
        gdf = gpd.read_file(path)
        gdf = gdf[gdf.geometry.notnull()]
        gdf = gdf[~gdf.geometry.is_empty]
        if ROOM_TYPE not in gdf.columns:
            gdf[ROOM_TYPE] = ""
        if ROOM_NAME not in gdf.columns:
            gdf[ROOM_NAME] = ""
        floor_gdfs[lvl] = gdf
        print(f"Loaded {len(gdf)} features from {path}")
    except Exception as e:
        print(f"Error reading {path}: {e}")
        floor_gdfs[lvl] = gpd.GeoDataFrame()

# --- HELPERS ---
def build_floor_graph(gdf):
    if gdf.empty or ROOM_TYPE not in gdf.columns:
        return nx.Graph()
    corridors = gdf[gdf[ROOM_TYPE].astype(str).str.contains("corridor", case=False, na=False)]
    G = nx.Graph()
    for geom in corridors.geometry:
        if geom is None or geom.is_empty:
            continue
        boundary = geom.boundary
        lines = [boundary] if boundary.geom_type == "LineString" else list(boundary.geoms)
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords)-1):
                p1 = Point(coords[i])
                p2 = Point(coords[i+1])
                G.add_edge((p1.x, p1.y), (p2.x, p2.y), weight=p1.distance(p2))
    return G

def connect_to_corridor(point, G):
    if G is None or len(G.nodes) == 0:
        return None
    nearest = None
    min_d = float("inf")
    pt = Point(point) if not isinstance(point, Point) else point
    for node in G.nodes:
        try:
            node_pt = Point(node)
            d = pt.distance(node_pt)
        except Exception:
            continue
        if d < min_d:
            min_d = d
            nearest = node
    return nearest

def nearest_pair_list(list_a, list_b):
    pairs = []
    used_b = set()
    for a in list_a:
        best_b = None
        best_d = float("inf")
        pa = Point(a)
        for b in list_b:
            if b in used_b:
                continue
            d = pa.distance(Point(b))
            if d < best_d:
                best_d = d
                best_b = b
        if best_b is not None:
            pairs.append((a, best_b))
            used_b.add(best_b)
    return pairs

# --- BUILD FLOOR GRAPHS & STAIRS ---
floor_graphs = {}
floor_stairs = {}
for lvl, gdf in floor_gdfs.items():
    G = build_floor_graph(gdf)
    floor_graphs[lvl] = G

    # Add staircase centroids
    stair_nodes = []
    if not gdf.empty:
        stairs = gdf[gdf[ROOM_TYPE].astype(str).str.contains("staircase", case=False, na=False)]
        for _, row in stairs.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            c = geom.centroid
            if c is None or c.is_empty:
                continue
            nearest = connect_to_corridor(c, G)
            if nearest:
                G.add_edge((float(c.x), float(c.y)), nearest, weight=0.5)
            stair_nodes.append((float(c.x), float(c.y)))
    floor_stairs[lvl] = stair_nodes

# --- SMART STAIR CONNECTIONS ---
# Match staircases by their name (SG01 <-> S201 etc.)
def get_stair_pairs():
    pairs = []
    levels = sorted(floor_gdfs.keys())
    for i in range(len(levels)-1):
        a, b = levels[i], levels[i+1]
        gdf_a, gdf_b = floor_gdfs[a], floor_gdfs[b]
        stairs_a = gdf_a[gdf_a[ROOM_TYPE].astype(str).str.contains("staircase", case=False, na=False)]
        stairs_b = gdf_b[gdf_b[ROOM_TYPE].astype(str).str.contains("staircase", case=False, na=False)]

        for _, row_a in stairs_a.iterrows():
            name_a = str(row_a.get(ROOM_NAME, "")).strip().lower()
            geom_a = row_a.geometry
            if geom_a is None or geom_a.is_empty:
                continue
            cent_a = (float(geom_a.centroid.x), float(geom_a.centroid.y))

            best_match = None
            min_dist = float("inf")
            for _, row_b in stairs_b.iterrows():
                name_b = str(row_b.get(ROOM_NAME, "")).strip().lower()
                geom_b = row_b.geometry
                if geom_b is None or geom_b.is_empty:
                    continue
                cent_b = (float(geom_b.centroid.x), float(geom_b.centroid.y))

                # match by similar ID (SG01 <-> S201 etc.)
                if name_a[:2] == name_b[:2] or name_a[-2:] == name_b[-2:]:
                    d = Point(cent_a).distance(Point(cent_b))
                    if d < min_dist:
                        min_dist = d
                        best_match = cent_b

            if best_match:
                pairs.append(((a, cent_a), (b, best_match)))
    return pairs

# --- MERGE GRAPH ---
G_all = nx.Graph()
for lvl, G in floor_graphs.items():
    for node in G.nodes:
        G_all.add_node((lvl, node))
    for u, v, data in G.edges(data=True):
        G_all.add_edge((lvl, u), (lvl, v), weight=data.get("weight", 1.0))

# --- Connect staircases between floors based on IDs ---
stair_pairs = get_stair_pairs()
for (a, sa), (b, sb) in stair_pairs:
    G_all.add_edge((a, sa), (b, sb), weight=1.0)

# --- ROOMS LIST ---
all_rooms = []
for lvl, gdf in floor_gdfs.items():
    if gdf.empty:
        continue
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        centroid = geom.centroid
        if centroid is None or centroid.is_empty:
            continue
        rtype = str(row.get(ROOM_TYPE, "")).strip()
        rname = str(row.get(ROOM_NAME, "")).strip()
        all_rooms.append({
            "floor": lvl,
            "room_type": rtype,
            "room_name": rname,
            "coords": (float(centroid.x), float(centroid.y))
        })

room_name_list = [r["room_name"] for r in all_rooms if r["room_name"]]
room_type_list = [r["room_type"] for r in all_rooms if r["room_type"]]
match_candidates = sorted(list(set(room_name_list + room_type_list)))

# --- MATCHING FUNCTIONS ---
def canonical(s):
    if s is None:
        return ""
    s2 = str(s).strip().lower()
    s_alnum = "".join(ch for ch in s2 if ch.isalnum())
    return s2, s_alnum

def find_best_match(input_str):
    if not input_str:
        return None, None, []

    q_raw, q_alnum = canonical(input_str)

    for r in all_rooms:
        if r["room_name"].strip().lower() == q_raw:
            floor = r["floor"]
            centroid = Point(r["coords"])
            node = connect_to_corridor(centroid, floor_graphs[floor])
            return (floor, node), centroid, [r["room_name"]]

    for r in all_rooms:
        if r["room_type"].strip().lower() == q_raw:
            floor = r["floor"]
            centroid = Point(r["coords"])
            node = connect_to_corridor(centroid, floor_graphs[floor])
            return (floor, node), centroid, [r["room_type"]]

    substr_matches = []
    for r in all_rooms:
        rn = r["room_name"].strip().lower()
        if q_raw and q_raw in rn:
            substr_matches.append(r)
    if substr_matches:
        r = substr_matches[0]
        node = connect_to_corridor(Point(r["coords"]), floor_graphs[r["floor"]])
        return (r["floor"], node), Point(r["coords"]), [x["room_name"] for x in substr_matches[:10]]

    if q_alnum:
        for r in all_rooms:
            rn_alnum = "".join(ch for ch in r["room_name"].strip().lower() if ch.isalnum())
            if rn_alnum and rn_alnum == q_alnum:
                node = connect_to_corridor(Point(r["coords"]), floor_graphs[r["floor"]])
                return (r["floor"], node), Point(r["coords"]), [r["room_name"]]

    for r in all_rooms:
        rt = r["room_type"].strip().lower()
        if q_raw and q_raw in rt:
            node = connect_to_corridor(Point(r["coords"]), floor_graphs[r["floor"]])
            return (r["floor"], node), Point(r["coords"]), [r["room_type"]]

    close = difflib.get_close_matches(input_str, match_candidates, n=5, cutoff=0.6)
    return None, None, close

# --- ROUTES ---
@app.route("/debug_rooms")
def debug_rooms():
    return jsonify({"count": len(all_rooms),
                    "room_names": room_name_list[:200],
                    "room_types": list(set(room_type_list))[:200],
                    "candidates_sample": match_candidates[:200]})

@app.route("/get_rooms")
def get_rooms():
    types = sorted(list({r["room_type"] for r in all_rooms if r["room_type"]}))
    return jsonify(types)

@app.route("/get_path", methods=["POST"])
def get_path():
    data = request.get_json() or {}
    start_in = data.get("start", "")
    end_in = data.get("end", "")

    start_node, start_centroid, start_candidates = find_best_match(start_in)
    if not start_node:
        return jsonify({
            "error": f"Start '{start_in}' not found",
            "candidates": start_candidates
        }), 400

    if isinstance(end_in, str) and end_in.strip() == "Exit":
        end_node, end_centroid = None, None
        exit_rooms = [r for r in all_rooms if "exit" in r["room_type"].lower() or "exit" in r["room_name"].lower()]
        best = None
        best_len = float("inf")
        best_cent = None
        for r in exit_rooms:
            enode = connect_to_corridor(Point(r["coords"]), floor_graphs[r["floor"]])
            if not enode:
                continue
            try:
                path = nx.astar_path(G_all, start_node, (r["floor"], enode), weight="weight")
                length = sum(Point(u[1]).distance(Point(v[1])) for u, v in zip(path, path[1:]))
                if length < best_len:
                    best_len = length
                    best = (r["floor"], enode)
                    best_cent = Point(r["coords"])
            except Exception:
                continue
        if not best:
            return jsonify({"error": "No reachable emergency exit"}), 400
        end_node, end_centroid = best, best_cent
    else:
        end_node, end_centroid, end_candidates = find_best_match(end_in)
        if not end_node:
            return jsonify({
                "error": f"End '{end_in}' not found",
                "candidates": end_candidates
            }), 400

    try:
        path_nodes = nx.astar_path(G_all, start_node, end_node, weight="weight")
    except nx.NetworkXNoPath:
        return jsonify({"error": "No path between given nodes"}), 400

    floor_paths = {}
    for f, node in path_nodes:
        floor_paths.setdefault(f, []).append(node)

    fig, axes = plt.subplots(len(floor_paths), 1, figsize=(10, 6 * len(floor_paths)))
    if len(floor_paths) == 1:
        axes = [axes]
    for ax, (floor, nodes) in zip(axes, floor_paths.items()):
        gdf = floor_gdfs.get(floor)
        if gdf is None or gdf.empty:
            ax.set_title(f"{floor} (no data)")
            continue

        gdf.plot(ax=ax, color="lightgrey", edgecolor="black")

        # --- DISPLAY ROOM NUMBERS ON POLYGONS ---
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            c = geom.centroid
            if c is None or c.is_empty:
                continue
            room_no = str(row.get(ROOM_NAME, "")).strip()
            if room_no:
                ax.text(
                    c.x, c.y, room_no,
                    fontsize=7,
                    ha="center", va="center",
                    color="black", weight="bold",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5),
                    zorder=6
                )

        # draw path
        for i in range(len(nodes) - 1):
            seg = LineString([nodes[i], nodes[i+1]])
            ax.plot(*seg.xy, linewidth=2, linestyle="--", color="blue", zorder=5)

        # draw stairs
        for sx, sy in floor_stairs.get(floor, []):
            ax.scatter(sx, sy, s=80, edgecolor="black", facecolor="yellow", zorder=6)

        # start and end
        if floor == start_node[0]:
            ax.scatter(start_centroid.x, start_centroid.y, s=100, color="green", zorder=8)
        if floor == end_node[0]:
            ax.scatter(end_centroid.x, end_centroid.y, s=100, color="blue", zorder=8)

        ax.set_title(f"Path on {floor}")
        ax.axis("off")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)


