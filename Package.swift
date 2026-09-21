// swift-tools-version: 6.0

import PackageDescription

let package = Package(
  name: "laya-mlx-swift",
  platforms: [.macOS(.v14)],
  products: [
    .library(name: "LayaMLX", targets: ["LayaMLX"])
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
    .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
  ],
  targets: [
    .target(
      name: "LayaMLX",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXFast", package: "mlx-swift"),
        .product(name: "Hub", package: "swift-transformers"),
        .product(name: "Tokenizers", package: "swift-transformers"),
      ]
    ),
    .testTarget(name: "LayaMLXTests", dependencies: ["LayaMLX"]),
  ]
)
