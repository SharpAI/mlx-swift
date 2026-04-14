import XCTest
import Foundation

final class ForkProtectionTests: XCTestCase {
    
    /// This test ensures that the SharpAI custom SSD streaming kernel
    /// (`moe_stream.metal`) is natively preserved in the generated metal code and sources.
    /// If an upstream merge resets the metal codegen generator, this will safely fail.
    func testSSDStreamerMetalCodegenExists() throws {
        // Look for the metal source in the parent package structure to verify the submodule holds our patches
        let fileManager = FileManager.default
        let packageRoot = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        
        let customKernelPath = packageRoot
            .appendingPathComponent("Source/Cmlx/mlx/mlx/backend/metal/kernels/moe_stream.metal")
            
        XCTAssertTrue(fileManager.fileExists(atPath: customKernelPath.path), 
                      "🚨 ALARM: The SharpAI custom ssd-stream metal kernel is missing! A blind upstream Apple merge erased moe_stream.metal! Do not merge this PR.")
    }

    /// This test verifies that the custom ThreadPool modification we added to load.cpp is still tracked.
    func testThreadPoolFeatureExists() throws {
        let fileManager = FileManager.default
        let packageRoot = URL(fileURLWithPath: #file).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        
        let threadPoolPath = packageRoot.appendingPathComponent("Source/Cmlx/mlx/mlx/threadpool.h")
        XCTAssertTrue(fileManager.fileExists(atPath: threadPoolPath.path), 
                      "🚨 ALARM: The SharpAI custom ThreadPool is missing! Did the upstream ml-explore merge drop the background loader?")
    }
}
