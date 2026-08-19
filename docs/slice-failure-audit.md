# Complete Slice failure audit

## Scope and denominator

The final DQ step-4 batched inventory contains 48 Slice test instances. A fresh focused baseline reproduced 46 failures and 2 passes (`Slice2D_DefaultAxes` and `Slice1D_String`). The 46 failing test instances contain 50 distinct Slice input sets because `EmptyDim` contains two Slice calls and `CoalesceDims` contains four.

All 46 failures contain a Slice v10 graph whose control tensors are runtime inputs. Forty-three test instances also execute an initializer form of the same controls; `OptionalAxesInputAloneMissing` and the two invalid-axis tests are runtime-only. The failing runtime form must therefore not be interpreted as evidence that the initializer form also failed.

## Layered result

At baseline, 44 success-expected tests stopped at `No explicit or implicit shapes were provided for dynamic shape inputs`. The two expected-failure tests stopped at the same profile gate and therefore returned the wrong error text. Supplying profiles test-by-test exposed the following next layer:

| Final category | Test instances | Input sets | Meaning |
|---|---:|---:|---|
| Fully executable after diagnostic profiles | 42 | 46 | TensorRT can execute these Slice forms in a prebuilt engine; sentinel cases used exact INT32-clamped profiles. |
| ORT/TensorRT semantic mismatch | 1 | 1 | `ReverseAllAxes_1` produces an empty TensorRT result for negative step plus `INT32_MAX` end. |
| Zero-extent negative Slice limitation | 1 | 1 of its 2 calls | The positive zero-extent call passes; the negative-step call fails engine construction. |
| Validation/error propagation | 2 | 2 | TensorRT finds the invalid axes, but the EP hides the detailed reason. |

Consequently, Slice still contributes zero tests proven to require runtime engine creation. It does, however, require more than a naive profile generator.

## What the profile experiments established

- One engine accepted changing `starts`, `ends`, positive and negative `steps`, and positive and negative `axes` when the values stayed inside the supplied ranges.
- A single `steps` range spanning negative to positive values worked; separate sign profiles are not required by this TensorRT-RTX build.
- Full INT32 bounds worked for ordinary values, negative/positive steps, negative axes, and moderate out-of-bounds values such as 1000.
- Ten input sets use INT64 sentinels. The profile parser itself uses `stoi`, so literal INT64 bounds are rejected. Exact INT32-clamped profiles passed, but a wide profile receiving raw `INT64_MIN/MAX` failed at execution with `shape calculation overflow`. Runtime sentinel normalization is therefore required for a general solution.
- Runtime values outside a profile are not rejected safely. An `ends=4` execution under an exact `ends=2` profile returned length 2, and runtime `axis=1` under an exact `axis=0` profile sliced axis 0. Runtime range validation is mandatory before relying on automatically generated bounds.
- Multi-axis broad bounds build, although TensorRT warns that duplicate-axis min/max corners are not self-consistent. Duplicate axes still require runtime validation.

## Input-pattern counts across the 50 failing input sets

| Pattern | Count |
|---|---:|
| out-of-bounds/clamping | 26 |
| default step=1 | 24 |
| negative step | 15 |
| INT64 sentinel | 10 |
| empty output | 10 |
| default axes | 5 |
| mixed step signs | 3 |
| ordinary bounds | 3 |
| INT32 sentinel | 2 |
| zero-extent input | 2 |
| duplicate axes | 1 |
| invalid axis | 1 |

## Complete per-input inventory

`P1` through `P5` correspond to the implementation phases proposed below.

| Test | Call | Data type/shape | starts | ends | axes | steps | Input form | Important traits | Final diagnosis |
|---|---:|---|---|---|---|---|---|---|---|
| `SliceTest.Slice1D_InvalidStartEndRange` (source line 100) | 1 | `float` [6] | `[3]` | `[2]` | `[0]` | `omitted` | runtime + initializer variants | default step=1, empty output | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice1D_ValidStartEndRange_NoOutput` (source line 111) | 1 | `float` [6] | `[2]` | `[2]` | `[0]` | `omitted` | runtime + initializer variants | default step=1, empty output | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice1D_Regular` (source line 122) | 1 | `float` [6] | `[2]` | `[4]` | `[0]` | `omitted` | runtime + initializer variants | default step=1 | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice1D_Perf` (source line 133) | 1 | `float` [1000] | `[2]` | `[502]` | `[0]` | `omitted` | runtime + initializer variants | default step=1 | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice1D_EndOutOfBounds` (source line 146) | 1 | `float` [6] | `[0]` | `[10]` | `omitted` | `omitted` | runtime + initializer variants | default axes, default step=1, out-of-bounds/clamping | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice1D_StartAndEndOutOfBounds` (source line 157) | 1 | `float` [6] | `[1000]` | `[1001]` | `omitted` | `omitted` | runtime + initializer variants | default axes, default step=1, empty output, out-of-bounds/clamping | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice2D_StartAndEndOutOfBounds` (source line 168) | 1 | `float` [2, 3] | `[0, 1000]` | `[10, 1000]` | `[0, 1]` | `omitted` | runtime + initializer variants | default step=1, empty output, out-of-bounds/clamping | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice2D_OneAxis` (source line 179) | 1 | `float` [6, 4] | `[1]` | `[3]` | `[0]` | `omitted` | runtime + initializer variants | default step=1 | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice2D_TwoAxes` (source line 196) | 1 | `float` [6, 4] | `[2, 3]` | `[1000, -1]` | `[1, 0]` | `omitted` | runtime + initializer variants | default step=1, out-of-bounds/clamping | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice2D_TwoAxesEque` (source line 213) | 1 | `float` [6, 4] | `[2, 3]` | `[1000, 3]` | `[1, 0]` | `omitted` | runtime + initializer variants | default step=1, empty output, out-of-bounds/clamping | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice3D` (source line 240) | 1 | `float` [3, 3, 3] | `[0, 1, 1]` | `[1000, 1000, 1000]` | `omitted` | `omitted` | runtime + initializer variants | default axes, default step=1, out-of-bounds/clamping | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice1D_Int32` (source line 281) | 1 | `int32_t` [6] | `[2]` | `[4]` | `[0]` | `omitted` | runtime + initializer variants | default step=1 | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice1D_Int64` (source line 285) | 1 | `int64_t` [6] | `[2]` | `[4]` | `[0]` | `omitted` | runtime + initializer variants | default step=1 | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice1D_Float` (source line 289) | 1 | `float` [6] | `[2]` | `[4]` | `[0]` | `omitted` | runtime + initializer variants | default step=1 | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice1D_Float16` (source line 293) | 1 | `MLFloat16` [6] | `[2]` | `[4]` | `[0]` | `omitted` | runtime + initializer variants | default step=1 | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice1D_WithNegativeSteps_Regular` (source line 331) | 1 | `float` [4] | `[-1]` | `[-4]` | `[0]` | `[-1]` | runtime + initializer variants | negative step | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice1D_WithNegativeSteps_EndOutOfBounds_1` (source line 343) | 1 | `float` [6] | `[0]` | `[6]` | `[0]` | `[-1]` | runtime + initializer variants | negative step, empty output | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice1D_WithNegativeSteps_EndOutOfBounds_2` (source line 355) | 1 | `float` [6] | `[0]` | `[-10]` | `[0]` | `[-1]` | runtime + initializer variants | negative step, out-of-bounds/clamping | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice1D_WithNegativeSteps_ValidStartEndRange` (source line 370) | 1 | `float` [6] | `[5]` | `[0]` | `[0]` | `[-1]` | runtime + initializer variants | negative step | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice1D_WithNegativeSteps_StartOutOfBounds` (source line 382) | 1 | `float` [6] | `[7]` | `[0]` | `[0]` | `[-3]` | runtime + initializer variants | negative step, out-of-bounds/clamping | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice2D_WithPositiveSteps_1` (source line 394) | 1 | `float` [2, 4] | `[1, 0]` | `[2, 3]` | `[0, 1]` | `[1, 2]` | runtime + initializer variants | ordinary bounds | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice2D_WithPositiveSteps_2` (source line 406) | 1 | `float` [2, 4] | `[0, 1]` | `[-1, 1000]` | `omitted` | `omitted` | runtime + initializer variants | default axes, default step=1, out-of-bounds/clamping | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice2D_WithNegativeSteps_1` (source line 418) | 1 | `float` [2, 4] | `[1, 0]` | `[2, 3]` | `[0, 1]` | `[-1, -2]` | runtime + initializer variants | negative step, empty output | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice2D_WithNegativeSteps_2` (source line 430) | 1 | `float` [2, 4] | `[1, 3]` | `[0, 1]` | `[0, 1]` | `[-1, -2]` | runtime + initializer variants | negative step | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice3D_WithPositiveSteps_AllAxes` (source line 442) | 1 | `int32_t` [3, 3, 3] | `[0, 1, 1]` | `[1000, 1000, 1000]` | `[0, 1, 2]` | `[2, 2, 2]` | runtime + initializer variants | out-of-bounds/clamping | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice3D_FlattenInnermostDimsIncopy` (source line 464) | 1 | `int32_t` [3, 3, 3] | `[0, 0, 1]` | `[1000, 1000, 1000]` | `[2, 1, 0]` | `[1, 1, 2]` | runtime + initializer variants | out-of-bounds/clamping | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice3D_WithPositiveAndNegativeSteps_SubsetOfAxes_1` (source line 488) | 1 | `int32_t` [3, 3, 3] | `[1, 4]` | `[1000, 1]` | `[1, 2]` | `[3, -2]` | runtime + initializer variants | mixed step signs, out-of-bounds/clamping | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice3D_WithPositiveAndNegativeSteps_SubsetOfAxes_2` (source line 510) | 1 | `int32_t` [3, 3, 3] | `[1, 4]` | `[1000, 2]` | `[1, 2]` | `[3, -2]` | runtime + initializer variants | mixed step signs, empty output, out-of-bounds/clamping | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice1D_ReverseAllAxes_1` (source line 535) | 1 | `float` [4] | `[-1]` | `[INT32_MAX]` | `[0]` | `[-1]` | runtime + initializer variants | negative step, INT32 sentinel, out-of-bounds/clamping | **P3 semantic mismatch** — Engine ran but returned shape [0] instead of ORT's expected full reversal [4]. |
| `SliceTest.Slice1D_ReverseAllAxes_2` (source line 556) | 1 | `float` [4] | `[-1]` | `[INT32_MIN]` | `[0]` | `[-1]` | runtime + initializer variants | negative step, INT32 sentinel, out-of-bounds/clamping | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice1D_ReverseAllAxes_3` (source line 569) | 1 | `float` [4] | `[-1]` | `[-5]` | `[0]` | `[-1]` | runtime + initializer variants | negative step | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.Slice2D_ReverseAllAxes` (source line 581) | 1 | `float` [2, 2] | `[-1, -1]` | `[INT64_MIN, INT64_MIN]` | `[0, 1]` | `[-1, -1]` | runtime + initializer variants | negative step, INT64 sentinel, out-of-bounds/clamping | **P2 sentinel normalization** — Passed with INT64 runtime sentinel and an INT32-clamped exact profile. |
| `SliceTest.Slice2D_ReverseSubsetOfAxes_1` (source line 598) | 1 | `float` [2, 2] | `[-1]` | `[INT64_MIN]` | `[1]` | `[-1]` | runtime + initializer variants | negative step, INT64 sentinel, out-of-bounds/clamping | **P2 sentinel normalization** — Passed with INT64 runtime sentinel and an INT32-clamped exact profile. |
| `SliceTest.Slice2D_ReverseSubsetOfAxes_2` (source line 615) | 1 | `float` [2, 2] | `[-1]` | `[INT64_MIN]` | `[0]` | `[-1]` | runtime + initializer variants | negative step, INT64 sentinel, out-of-bounds/clamping | **P2 sentinel normalization** — Passed with INT64 runtime sentinel and an INT32-clamped exact profile. |
| `SliceTest.Slice2D_ImplicitCopyBySlicingADimensionFully` (source line 633) | 1 | `float` [2, 2] | `[0]` | `[INT64_MAX]` | `[1]` | `[1]` | runtime + initializer variants | INT64 sentinel, out-of-bounds/clamping | **P2 sentinel normalization** — Passed with INT64 runtime sentinel and an INT32-clamped exact profile. |
| `SliceTest.OptionalAxesInputAloneMissing` (source line 645) | 1 | `float` [6] | `[2]` | `[4]` | `omitted` | `[1]` | runtime only | default axes | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.InvalidAxesOutOfBounds` (source line 669) | 1 | `float` [2, 2] | `[0]` | `[1]` | `[2]` | `omitted` | runtime only | default step=1, invalid axis | **P5 validation/error propagation** — TensorRT detected axis 2 for rank 2, but EP returned only generic engine-build failure. |
| `SliceTest.InvalidAxesDuplicates` (source line 681) | 1 | `float` [2, 2] | `[0, 0]` | `[1, 1]` | `[0, 0]` | `omitted` | runtime only | default step=1, duplicate axes | **P5 validation/error propagation** — TensorRT detected duplicate axis 0, but EP returned only generic engine-build failure. |
| `SliceTest.Slice2D_ReverseSubsetOfNegAxes_1` (source line 693) | 1 | `float` [2, 2] | `[-1]` | `[INT64_MIN]` | `[-1]` | `[-1]` | runtime + initializer variants | negative step, INT64 sentinel, out-of-bounds/clamping | **P2 sentinel normalization** — Passed with INT64 runtime sentinel and an INT32-clamped exact profile. |
| `SliceTest.Slice5D_SubsetOfAxes_Flatten2Dims_OffsetInput` (source line 712) | 1 | `float` [1, 2, 2, 2, 2] | `[0, 1, 1, 0]` | `[1, 2, INT64_MAX, 6]` | `[0, 1, 2, 3]` | `omitted` | runtime + initializer variants | default step=1, INT64 sentinel, out-of-bounds/clamping | **P2 sentinel normalization** — Passed with INT64 runtime sentinel and an INT32-clamped exact profile. |
| `SliceTest.Slice5D_LargeStep` (source line 730) | 1 | `float` [1, 2, 2, 2, 2] | `[0]` | `[1]` | `[1]` | `[INT64_MAX]` | runtime + initializer variants | INT64 sentinel | **P2 sentinel normalization** — Passed with INT64 runtime sentinel and an INT32-clamped exact profile. |
| `SliceTest.Slice5D_CopyAxis2LargeBlock` (source line 747) | 1 | `float` [1, 3, 4, 2, 2] | `[0, 1]` | `[2, 3]` | `[1, 2]` | `omitted` | runtime + initializer variants | default step=1 | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest.EmptyDim` (source line 778) | 1 | `float` [0, 6] | `[0]` | `[1]` | `[0]` | `omitted` | runtime + initializer variants | default step=1, zero-extent input, empty output, out-of-bounds/clamping | **P1 exact profile** — Positive-step zero-extent case passed in isolation with an exact profile. |
| `SliceTest.EmptyDim` (source line 778) | 2 | `float` [0, 6] | `[1]` | `[0]` | `[0]` | `[-1]` | runtime + initializer variants | negative step, zero-extent input, empty output, out-of-bounds/clamping | **P4 zero-extent negative Slice** — TensorRT engine build failed: starts operand cannot be negative at axis 0. |
| `SliceTest.CoalesceDims` (source line 798) | 1 | `float` [2, 2, 2, 2] | `[1, 1]` | `[0, 2]` | `[0, 1]` | `[-1, 1]` | runtime + initializer variants | mixed step signs | **P1 exact profile** — Passed in an isolated model with its exact profile. |
| `SliceTest.CoalesceDims` (source line 798) | 2 | `float` [1, 2, 2, 2, 2] | `[1]` | `[INT64_MAX]` | `[2]` | `omitted` | runtime + initializer variants | default step=1, INT64 sentinel, out-of-bounds/clamping | **P2 sentinel normalization** — Passed in isolation with INT64_MAX at runtime and INT32_MAX in the profile. |
| `SliceTest.CoalesceDims` (source line 798) | 3 | `float` [1, 2, 2, 2, 2] | `[1, 1]` | `[INT64_MAX, INT64_MAX]` | `[1, 3]` | `omitted` | runtime + initializer variants | default step=1, INT64 sentinel, out-of-bounds/clamping | **P2 sentinel normalization** — Passed in isolation with INT64_MAX at runtime and INT32_MAX in the profile. |
| `SliceTest.CoalesceDims` (source line 798) | 4 | `float` [1, 1, 1] | `[0]` | `[INT64_MAX]` | `[1]` | `omitted` | runtime + initializer variants | default step=1, INT64 sentinel, out-of-bounds/clamping | **P2 sentinel normalization** — Passed in isolation with INT64_MAX at runtime and INT32_MAX in the profile. |
| `SliceTest/0.Slice1D_WithPositiveSteps` (source line 316) | 1 | `float` [6] | `[0]` | `[6]` | `[0]` | `[2]` | runtime + initializer variants | ordinary bounds | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |
| `SliceTest/1.Slice1D_WithPositiveSteps` (source line 316) | 1 | `MLFloat16` [6] | `[0]` | `[6]` | `[0]` | `[2]` | runtime + initializer variants | ordinary bounds | **P1 exact profile** — Passed after exact shape-value profiles were supplied. |

## Proposed implementation order

### P0: runtime shape-value range validation

Before adding guessed or automatic bounds, compare every runtime shape-tensor value with the active profile. Reject an out-of-range value instead of silently executing an engine specialized for another value. This is a general EP correctness fix, not only a Slice fix.

### P1: Slice-aware profiles for ordinary INT32-domain values

Identify Slice control inputs by position. Use the known control-vector length, valid axis range `[-rank, rank-1]`, valid optimum axes, nonzero optimum steps, and tested INT32-domain bounds for starts, ends, and steps. Preserve explicit user profiles as the override. This should address the ordinary profile-only group, but must be rerun against all 50 input sets.

### P2: runtime normalization of Slice INT64 sentinels

Clamp Slice `starts`, `ends`, and very large `steps` to the TensorRT INT32 shape domain before the Slice layer consumes them. This must be a runtime value transformation or equivalent graph preprocessing; changing only the profile bounds does not prevent the observed `INT64_MIN/MAX` overflow.

### P3: decide the negative-step plus `INT32_MAX` semantic policy

`Slice1D_ReverseAllAxes_1` expects ORT's full-reversal behavior, while TensorRT returns an empty slice. Confirm the desired ONNX/ORT compatibility policy with the mentor. Options are a targeted conditional normalization of `ends` when `steps < 0`, declining this case, or accepting TensorRT semantics if the test expectation is intentionally not part of EP compatibility.

### P4: zero-extent negative Slice

Do not classify all zero-extent Slice inputs as unsupported: the positive-step case passed. For the negative-step case, either add an EP-side empty-output short circuit or decline only this form until TensorRT supports it.

### P5: runtime axes validation and error preservation

Validate axis range and uniqueness before TensorRT execution and return the detailed ORT-compatible reason. Also preserve TensorRT build diagnostics instead of replacing them with only `Failed to create serialized engine`.

## Evidence files

The reproducible scripts, generated models, XML, and per-test logs are under the untracked directory `build-phase1-ninja/slice-full-audit`. The authoritative test inputs are in the adjacent ORT checkout at `onnxruntime/test/providers/cpu/tensor/slice_op.test.cc`.
