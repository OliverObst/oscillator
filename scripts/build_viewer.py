"""Build static viewer assets from a local checkpoint and the pinned upstream K1 model."""

import argparse
import json
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from oscillator.web_export import export_web

UPSTREAM_REVISION = "d92e11395d961096d639446c0a8a176f12bb51da"
UPSTREAM = (
    f"https://raw.githubusercontent.com/IntelligentRoboticsLab/booster_mjlab/{UPSTREAM_REVISION}"
)
ROOT = Path(__file__).resolve().parents[1]


def download_asset(name, directory):
    path = directory / name
    if not path.exists():
        with urllib.request.urlopen(f"{UPSTREAM}/website/static/demo/{name}", timeout=30) as r:
            path.write_bytes(r.read())
    return path


def skeleton(xml, scene):
    def numbers(text, default):
        return [float(x) for x in text.split()] if text else default

    result = []
    joint_names = [j["name"] for j in scene["joints"]]

    def visit(body, parent):
        index = len(result)
        joints = body.findall("joint")
        joint = next((j for j in joints if j.get("type", "hinge") != "free"), None)
        if len(joints) > 1 or (joint is not None and joint.get("type", "hinge") != "hinge"):
            raise ValueError("Viewer supports one hinge per body in the published K1 model")
        result.append(
            {
                "name": body.get("name"),
                "parent": parent,
                "position": numbers(body.get("pos"), [0, 0, 0]),
                "quaternion_wxyz": numbers(body.get("quat"), [1, 0, 0, 0]),
                "joint": None if joint is None else joint_names.index(joint.get("name")),
                "axis": [0, 0, 1] if joint is None else numbers(joint.get("axis"), [0, 0, 1]),
                "pivot": [0, 0, 0] if joint is None else numbers(joint.get("pos"), [0, 0, 0]),
            }
        )
        for child in body.findall("body"):
            visit(child, index)

    visit(ET.parse(xml).find("worldbody/body"), -1)
    names = [body["name"] for body in result]
    return {
        "bodies": result,
        "joint_names": joint_names,
        "meshes": [
            {"node": n["node"], "body": names.index(n["body"])} for n in scene["render"]["nodes"]
        ],
        "feet": [names.index("left_foot_link"), names.index("right_foot_link")],
        "upstream_revision": UPSTREAM_REVISION,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/baseline/model.pt")
    args = parser.parse_args()
    output = ROOT / "web/assets"
    output.mkdir(parents=True, exist_ok=True)
    manifest = export_web(args.data, args.checkpoint, output)
    xml = download_asset("k1_web.xml", output)
    scene = json.loads(download_asset("scene.json", output).read_text())
    download_asset("robot.glb", output)
    (output / "skeleton.json").write_text(json.dumps(skeleton(xml, scene), indent=2) + "\n")
    # Vendor only the modules the viewer needs, making playback independent of CDNs.
    package = ROOT / "node_modules/three"
    vendor = ROOT / "web/vendor/three"
    files = [
        "LICENSE",
        "build/three.module.js",
        "build/three.core.js",
        "examples/jsm/loaders/GLTFLoader.js",
        "examples/jsm/loaders/DRACOLoader.js",
        "examples/jsm/controls/OrbitControls.js",
        "examples/jsm/utils/BufferGeometryUtils.js",
        "examples/jsm/utils/SkeletonUtils.js",
    ]
    files += [
        str(p.relative_to(package))
        for p in (package / "examples/jsm/libs/draco/gltf").iterdir()
        if p.is_file()
    ]
    for relative in files:
        dest = vendor / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / relative, dest)
    print(f"Exported {len(manifest['clips'])} source clips, decoder and K1 meshes to web/")


if __name__ == "__main__":
    main()
