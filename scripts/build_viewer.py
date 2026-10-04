"""Build static viewer assets from a local checkpoint and the pinned upstream K1 model."""

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from oscillator.comparison_report import write_comparison_pages
from oscillator.preprocess import read_prepared
from oscillator.train import evaluate, load_checkpoint
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
    parser.add_argument("--comparison", type=Path, help="Development-selected comparison.json")
    args = parser.parse_args()
    output = ROOT / "web/assets"
    output.mkdir(parents=True, exist_ok=True)
    comparison = json.loads(args.comparison.read_text()) if args.comparison else None
    checkpoint = args.checkpoint
    if comparison and not checkpoint.exists():
        checkpoint = Path(
            next(
                c["checkpoint"]
                for c in comparison["groups"]["harmonic3"]["candidates"]
                if c["seed"] == 7
            )
        )
    manifest = export_web(args.data, checkpoint, output)
    if comparison:
        models = []
        for key, group in comparison["groups"].items():
            chosen = next(c for c in group["candidates"] if c["seed"] == group["selected_seed"])
            if key == "harmonic3":
                # Retain the existing published baseline; experimental seed means still
                # include all three runs. Its metrics come from the preserved checkpoint.
                chosen = next(c for c in group["candidates"] if c["seed"] == 7)
            model_name = "model" if key == "harmonic3" else key
            validation_name = "validation" if key == "harmonic3" else f"{key}-validation"
            exported = (
                manifest
                if key == "harmonic3"
                else export_web(
                    args.data,
                    Path(chosen["checkpoint"]),
                    output,
                    model_name=model_name,
                    validation_name=validation_name,
                    write_clips=False,
                )
            )
            if key == "harmonic3":
                # Derive foot diagnostics for the preserved baseline itself.
                rig = json.loads((output / "skeleton.json").read_text())
                metadata, clips = read_prepared(args.data)
                baseline, cadence, _ = load_checkpoint(checkpoint)
                metrics = {
                    split: evaluate(baseline, clips, names, cadence, skeleton=rig)
                    for split, names in metadata["splits"].items()
                }
                for clip in exported["clips"]:
                    clip["metrics"] = metrics[clip["split"]][clip["name"]]
                (output / "manifest.json").write_text(json.dumps(exported, indent=2) + "\n")
            models.append(
                {
                    "id": key,
                    "label": group["label"],
                    "file": f"{model_name}.json",
                    "validation": f"{validation_name}.json",
                    "seed": chosen["seed"],
                    "parameters": chosen["parameters"],
                    "tau_z_s": chosen["tau_z_s"],
                    "selection_note": "original baseline"
                    if key == "harmonic3"
                    else "development-selected",
                    "metrics": {c["name"]: c["metrics"] for c in exported["clips"]},
                    "test_joint_rmse_mean_rad": group["test_joint_rmse_mean_rad"],
                    "test_joint_rmse_range_rad": group["test_joint_rmse_range_rad"],
                }
            )
        (output / "models.json").write_text(
            json.dumps({"models": models, "protocol": comparison["protocol"]}, indent=2) + "\n"
        )
        # Publish a portable report without machine-specific checkpoint locations.
        public_report = json.loads(json.dumps(comparison))
        for group in public_report["groups"].values():
            group.pop("selected_checkpoint")
            for candidate in group["candidates"]:
                candidate.pop("checkpoint")
        (output / "comparison.json").write_text(json.dumps(public_report, indent=2) + "\n")
        write_comparison_pages(public_report, ROOT)
    else:
        (output / "models.json").write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "id": "harmonic3",
                            "label": "Harmonic decoder",
                            "file": "model.json",
                            "validation": "validation.json",
                            "metrics": {c["name"]: c["metrics"] for c in manifest["clips"]},
                        }
                    ]
                },
                indent=2,
            )
            + "\n"
        )
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
    # Content versions also refresh cached ES modules when a previously open viewer
    # reloads after deployment. Core must be versioned before hashing its importer.
    web = ROOT / "web"
    core_version = hashlib.sha256((web / "core.js").read_bytes()).hexdigest()[:12]
    app = web / "app.js"
    app.write_text(
        re.sub(r"\./core\.js(?:\?v=[a-z0-9]+)?", f"./core.js?v={core_version}", app.read_text())
    )
    app_version = hashlib.sha256(app.read_bytes()).hexdigest()[:12]
    style_version = hashlib.sha256((web / "style.css").read_bytes()).hexdigest()[:12]
    for page in web.glob("*.html"):
        text = re.sub(
            r"\./app\.js(?:\?v=[a-z0-9]+)?", f"./app.js?v={app_version}", page.read_text()
        )
        page.write_text(
            re.sub(r"\./style\.css(?:\?v=[a-z0-9]+)?", f"./style.css?v={style_version}", text)
        )
    print(f"Exported {len(manifest['clips'])} source clips, decoder and K1 meshes to web/")


if __name__ == "__main__":
    main()
