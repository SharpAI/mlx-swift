// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

class FloorDivideTests: XCTestCase {

    /// mlx #4515: integer floor_divide rounds toward negative infinity like numpy.
    func testFloorDivideSignedIntegersRoundsTowardNegativeInfinity() {
        let a = MLXArray([4, 5, -1, -6] as [Int32])
        let b = MLXArray([-2, 3, 2, -3] as [Int32])
        XCTAssertEqual(a.floorDivide(b).asArray(Int32.self), [-2, 1, -1, 2])
    }

    /// The MoE expert sort divides uint32 sort indices by top-k. Unsigned values need no sign
    /// correction, so the result must match plain integer division.
    func testFloorDivideUnsignedMatchesIntegerDivision() {
        let order = MLXArray((0 ..< 64).map { UInt32($0) })
        let q = order.floorDivide(8)
        XCTAssertEqual(q.dtype, .uint32)
        XCTAssertEqual(q.asArray(UInt32.self), (0 ..< 64).map { UInt32($0 / 8) })
    }
}
