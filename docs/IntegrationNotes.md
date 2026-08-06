# PR 27-29 Reconstruction Notes

## Immutable upstream points

| State | Commit |
| --- | --- |
| PR #27 merged baseline | `1a624a9` |
| PR #28 original head | `8f73a3b` |
| PR #28 historical upstream merge | `6538695` |
| PR #29 original head | `d739396` |
| Corrected PR #29-first fork base | `b4c5a24` |

PR #28 and PR #29 were independently authored from the PR #27 baseline. The
historical upstream repository merged PR #28 first, which left PR #29
conflict-dirty. The corrected fork intentionally uses the opposite order.

## Corrected reconstruction procedure

1. Reset the integration base to the PR #27 merge, `1a624a9`.
2. Merge the original PR #29 head, `d739396`, with a merge commit.
3. Verify that the resulting base tree is byte-for-byte identical to PR #29's
   head. This ensures the rendering fix is not rewritten by the reconstruction.
4. Create the PR #28-plus feature branch from that PR #29-first base.
5. Port PR #28's thermal-body, thermal-solver, validation, sensor-configuration,
   and IR-camera changes onto the new base.
6. Resolve overlaps by retaining PR #29's independent rendering corrections and
   adapting PR #28 around them:
   - Far clipping range correction
   - Correct `orientation` configuration spelling
   - Removal of duplicate agent position/orientation initialization
   - EO camera `camera_offset` and `camera_angle` behavior
7. Keep PR #28's configurable `mount_position` and `mount_hpr` interface while
   accepting PR #29's mount names as compatibility fallbacks.
8. Replay the later live-temperature rendering, drone controls, GPU thermal
   visualization, target drone, and temperature-guide commits.
9. Apply code-quality and documentation changes as a separate final commit.

## Conflict policy

- The corrected base preserves every file from PR #29, including
  `logs/buffer/buffer.ppm`; the PR #28-plus branch does not modify that artifact.
- PR #29 remains authoritative for rendering, transform, and EO camera fixes.
- PR #28 remains authoritative for the thermal equations, material values,
  sanity artifacts, `ThermalBody` behavior, and configurable IR sensor model.
- Shared loaders expose compatibility fields instead of duplicating PR #28 and
  PR #29 setup paths.
- Refactoring is isolated from feature commits and preserves the implemented
  thermal equations and detector behavior.

This history makes the dependency explicit: PR #29 is already present in the
base, while the pull request contains only the adjusted PR #28 work and later
features needed above it.
