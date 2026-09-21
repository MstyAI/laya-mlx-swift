# Contributing

Issues and focused pull requests are welcome.

Before opening a pull request:

```bash
swift build
xcodebuild \
  -scheme laya-mlx-swift \
  -destination 'platform=macOS,arch=arm64' \
  -skipPackagePluginValidation \
  test
```

Keep public API changes small and include tests for prompt formatting, model loading, or output behavior when those contracts change.
