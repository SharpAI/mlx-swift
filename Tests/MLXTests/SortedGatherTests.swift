// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

/// `gatherMM` / `gatherQuantizedMM` with `sortedIndices: true` take the "rhs sorted" Metal paths, which
/// call the ahead-of-time kernel `gather_mm_offsets` (mlx >= 0.32.3, #4567 / #4572). SwiftPM builds
/// `default.metallib` only from the kernels listed in `tools/fix-metal-includes.sh`, so a kernel that is
/// missing from that list only fails at run time, and only when this path is hit.
///
/// All inputs are small integers, so every sum is exact in float16, bfloat16, float32 and TF32, and the
/// Metal results can be compared with `arrayEqual` against a CPU reference. There is no tolerance to tune.
class SortedGatherTests: XCTestCase {

    /// Deterministic integers in `[-2, 2]`.
    private func smallIntegers(_ count: Int, seed: Int) -> [Float] {
        (0 ..< count).map { Float((($0 &* 7 &+ seed &* 13) % 5) - 2) }
    }

    /// Ascending expert ids. The group sizes are uneven and expert 1 is empty, which exercises the
    /// binary search in `gather_mm_offsets` for a group with no rows.
    private func sortedExpertIds(_ counts: [Int]) -> [UInt32] {
        counts.enumerated().flatMap { Array(repeating: UInt32($0.offset), count: $0.element) }
    }

    func testSortedGatherMMMatchesCPUReference() {
        let counts = [30, 0, 20, 14]
        let (batch, experts, k, n) = (counts.reduce(0, +), counts.count, 32, 32)
        let ids = MLXArray(sortedExpertIds(counts))

        let a32 = MLXArray(smallIntegers(batch * k, seed: 1), [batch, 1, k])
        let b32 = MLXArray(smallIntegers(experts * k * n, seed: 2), [experts, k, n])

        let expected = matmul(a32, b32.take(ids, axis: 0, stream: .cpu), stream: .cpu)

        for dtype in [DType.float16, .bfloat16, .float32] {
            let a = a32.asType(dtype)
            let b = b32.asType(dtype)

            let sorted = gatherMM(a, b, rhsIndices: ids, sortedIndices: true, stream: .gpu)
            let unsorted = gatherMM(a, b, rhsIndices: ids, sortedIndices: false, stream: .gpu)

            XCTAssertEqual(sorted.shape, [batch, 1, n], "\(dtype)")
            XCTAssertTrue(
                arrayEqual(sorted.asType(.float32), expected, stream: .cpu).item(Bool.self),
                "sorted gatherMM differs from the CPU reference (\(dtype))")
            XCTAssertTrue(
                arrayEqual(unsorted.asType(.float32), expected, stream: .cpu).item(Bool.self),
                "unsorted gatherMM differs from the CPU reference (\(dtype))")
        }
    }

    func testSortedGatherQuantizedMMMatchesCPUReference() {
        let counts = [30, 0, 20, 14]
        let (batch, experts, k, n) = (counts.reduce(0, +), counts.count, 64, 64)
        let ids = MLXArray(sortedExpertIds(counts))

        // Each quantization group (one row of `k == groupSize` values) holds the integers 0...15 and
        // contains both 0 and 15, so the affine scale is exactly 1 and the bias exactly 0.
        let weightValues = (0 ..< experts * n).flatMap { row in
            (0 ..< k).map { col -> Float in
                switch col {
                case 0: return 0
                case 1: return 15
                default: return Float((col &* 7 &+ row &* 3) % 16)
                }
            }
        }
        let w = MLXArray(weightValues, [experts, n, k]).asType(.float16)
        let (wq, scales, biases) = quantized(w, groupSize: 64, bits: 4)

        // x values in [-1, 1]: |sum| <= 64 * 15, exact in float16.
        let x32 = MLXArray(smallIntegers(batch * k, seed: 3).map { max(-1, min(1, $0)) }, [batch, 1, k])
        let x = x32.asType(.float16)

        let dequantizedWeights = dequantized(
            wq, scales: scales, biases: biases, groupSize: 64, bits: 4, dtype: .float32)
        let expected = matmul(
            x32, dequantizedWeights.take(ids, axis: 0, stream: .cpu).transposed(0, 2, 1, stream: .cpu),
            stream: .cpu)

        let sorted = gatherQuantizedMM(
            x, wq, scales: scales, biases: biases, rhsIndices: ids, transpose: true,
            groupSize: 64, bits: 4, sortedIndices: true, stream: .gpu)
        let unsorted = gatherQuantizedMM(
            x, wq, scales: scales, biases: biases, rhsIndices: ids, transpose: true,
            groupSize: 64, bits: 4, sortedIndices: false, stream: .gpu)

        XCTAssertEqual(sorted.shape, [batch, 1, n])
        XCTAssertTrue(
            arrayEqual(sorted.asType(.float32), expected, stream: .cpu).item(Bool.self),
            "sorted gatherQuantizedMM differs from the CPU reference")
        XCTAssertTrue(
            arrayEqual(unsorted.asType(.float32), expected, stream: .cpu).item(Bool.self),
            "unsorted gatherQuantizedMM differs from the CPU reference")
    }
}
