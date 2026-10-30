# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Add PhysX mimic coupling to the locally attached OnRobot RG2."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Sdf, Usd


FOLLOWER_REFERENCES = {
    "/World/org2/onrobot_rg2_base_link/left_inner_knuckle_joint": (
        "/World/org2/onrobot_rg2_base_link/finger_joint",
        1.0,
    ),
    "/World/org2/onrobot_rg2_base_link/right_inner_knuckle_joint": (
        "/World/org2/onrobot_rg2_base_link/finger_joint",
        1.0,
    ),
    "/World/org2/onrobot_rg2_base_link/right_outer_knuckle_joint": (
        "/World/org2/onrobot_rg2_base_link/finger_joint",
        -1.0,
    ),
    "/World/org2/left_outer_knuckle/left_inner_finger_joint": (
        "/World/org2/onrobot_rg2_base_link/finger_joint",
        1.0,
    ),
    "/World/org2/right_outer_knuckle/right_inner_finger_joint": (
        "/World/org2/onrobot_rg2_base_link/right_outer_knuckle_joint",
        1.0,
    ),
}


def add_api_schemas(prim: Usd.Prim, schemas: list[str]) -> None:
    """Preserve existing API schemas while prepending the requested schemas."""
    operation = prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
    existing = list(operation.prependedItems) + list(operation.explicitItems)
    operation.prependedItems = []
    operation.explicitItems = list(schemas) + [schema for schema in existing if schema not in schemas]
    prim.SetMetadata("apiSchemas", operation)


def create_float_attribute(prim: Usd.Prim, name: str, value: float) -> None:
    """Create or update a float attribute on a joint prim."""
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)


def repair(source: Path, target: Path) -> None:
    """Create a coupled RG2 USD copy from the attached UR5e source."""
    stage = Usd.Stage.Open(str(source))
    assert stage, f"Unable to open source USD: {source}"

    driver_path = Sdf.Path("/World/org2/onrobot_rg2_base_link/finger_joint")
    driver = stage.GetPrimAtPath(driver_path)
    assert driver.IsValid(), f"Missing RG2 driver joint: {driver_path}"

    add_api_schemas(driver, ["PhysxJointAPI"])
    driver.GetAttribute("drive:angular:physics:stiffness").Set(500.0)
    driver.GetAttribute("drive:angular:physics:damping").Set(50.0)
    driver.GetAttribute("drive:angular:physics:maxForce").Set(40.0)

    for follower_path_string, (reference_path_string, gearing) in FOLLOWER_REFERENCES.items():
        follower_path = Sdf.Path(follower_path_string)
        reference_path = Sdf.Path(reference_path_string)
        follower = stage.GetPrimAtPath(follower_path)
        assert follower.IsValid(), f"Missing RG2 follower joint: {follower_path}"

        add_api_schemas(follower, ["PhysxJointAPI", "PhysxMimicJointAPI:rotX"])
        create_float_attribute(follower, "physxMimicJoint:rotX:dampingRatio", 0.0)
        create_float_attribute(follower, "physxMimicJoint:rotX:naturalFrequency", 0.0)
        create_float_attribute(follower, "physxMimicJoint:rotX:gearing", gearing)
        create_float_attribute(follower, "physxMimicJoint:rotX:offset", 0.0)
        follower.CreateRelationship("physxMimicJoint:rotX:referenceJoint").SetTargets([reference_path])

        follower.GetAttribute("drive:angular:physics:stiffness").Set(0.0)
        follower.GetAttribute("drive:angular:physics:damping").Set(0.0)
        follower.GetAttribute("drive:angular:physics:maxForce").Set(0.0)

    target.parent.mkdir(parents=True, exist_ok=True)
    assert stage.GetRootLayer().Export(str(target)), f"Unable to write target USD: {target}"

    repaired = Usd.Stage.Open(str(target))
    assert repaired, f"Unable to reopen target USD: {target}"
    repaired_driver = repaired.GetPrimAtPath(driver_path)
    for follower_path_string, (reference_path_string, gearing) in FOLLOWER_REFERENCES.items():
        follower_path = Sdf.Path(follower_path_string)
        reference_path = Sdf.Path(reference_path_string)
        follower = repaired.GetPrimAtPath(follower_path)
        schemas = list(follower.GetMetadata("apiSchemas").explicitItems)
        assert "PhysxJointAPI" in schemas
        assert "PhysxMimicJointAPI:rotX" in schemas
        assert list(follower.GetRelationship("physxMimicJoint:rotX:referenceJoint").GetTargets()) == [reference_path]
        assert follower.GetAttribute("physxMimicJoint:rotX:gearing").Get() == gearing
        assert follower.GetAttribute("drive:angular:physics:maxForce").Get() == 0.0

    print(f"Created and validated: {target}")
    print(f"Driver: {repaired_driver.GetPath()}")
    print(f"Followers: {len(FOLLOWER_REFERENCES)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    repair(args.source, args.target)


if __name__ == "__main__":
    main()
