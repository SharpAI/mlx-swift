sed -i '' -E 's|\.package\(url: "https://github.com/SharpAI/mlx-swift\.git".*|\.package(path: "../mlx-swift"),|' SwiftLM/Package.swift
sed -i '' -E 's|\.package\(url: "https://github.com/ml-explore/mlx-swift\.git".*|\.package(path: "../mlx-swift"),|' SwiftLM/Package.swift

sed -i '' -E 's|\.package\(url: "https://github.com/SharpAI/mlx-swift\.git".*|\.package(path: "../../mlx-swift"),|' SwiftLM/mlx-swift-lm/Package.swift
sed -i '' -E 's|\.package\(url: "https://github.com/ml-explore/mlx-swift\.git".*|\.package(path: "../../mlx-swift"),|' SwiftLM/mlx-swift-lm/Package.swift
