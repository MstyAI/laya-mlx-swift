# Laya MLX Swift

Run [Laya](https://github.com/NandhaKishorM/laya) directly inside a macOS app with Apple MLX.

- Native Swift library
- Runs locally on Apple silicon
- No Python, ONNX, helper process, or local server
- Downloads the model only when your app calls `prepare`

## Add the package

In Xcode, add:

```text
https://github.com/MstyAI/laya-mlx-swift
```

Then add `LayaMLX` to your target.

## Prepare once

Call this from a button or setup screen. The download is about 804 MiB.

```swift
import LayaMLX

let modelDirectory = try await LayaModel.prepare { progress in
    print("Downloaded \(Int(progress.fractionCompleted * 100))%")
}
```

`prepare` does nothing when the model is already present. Inference never starts a download.

## Make a decision

```swift
let laya = try await LayaAgent(modelDirectory: modelDirectory)

let result = try await laya.predict(LayaRequest(
    state: "The payment was declined twice and the customer is angry.",
    questions: [
        "route": .choice(
            instructions: "Choose the best destination.",
            options: [
                LayaOption("billing", description: "Payment and invoice problems"),
                LayaOption("technical", description: "Product bugs and outages"),
                LayaOption("sales", description: "Purchasing questions"),
            ]
        ),
        "urgent": .boolean(
            instructions: "This needs urgent human attention."
        ),
    ]
)))

print(result.answers["route"]?.choice ?? "unknown")
print(result.answers["urgent"]?.booleanProbability ?? 0)
```

Laya supports three question types:

| Type | Use it for | Result |
|---|---|---|
| `choice` | Routing or selecting an action | Winning label and probabilities |
| `boolean` | Checking whether a statement holds | Probability from 0 to 1 |
| `score` | Rating against an ordered scale | Weighted score and probabilities |

## Performance

On an M3 Max, one request with three decisions takes **27.1 ms p50** and **28.2 ms p95**. Python MLX takes 27.0 ms and our ONNX port takes 397 ms on the same benchmark. Accuracy matches both Laya ports exactly.

[See the benchmark method and raw results](BENCHMARKS.md).

## Requirements

- Apple silicon Mac
- macOS 14 or newer
- Xcode 26 or newer

MLX compiles Metal shaders as part of the Xcode build. `swift build` can check the Swift code, but use Xcode or `xcodebuild` to build and test an app that runs inference.

<details>
<summary>Use your own model directory</summary>

The directory must contain:

```text
model.safetensors
rl_agent_config.json
encoder/config.json
tokenizer/tokenizer.json
tokenizer/tokenizer_config.json
```

Pass it directly to `LayaAgent`:

```swift
let laya = try await LayaAgent(modelDirectory: modelDirectory)
```

</details>

<details>
<summary>Build and test</summary>

```bash
swift build
xcodebuild \
  -scheme laya-mlx-swift \
  -destination 'platform=macOS,arch=arm64' \
  -skipPackagePluginValidation \
  test
```

Checkpoint tests use the prepared default model or the standard Hugging Face cache location.

</details>

## License

Apache 2.0. Laya and model weights have their own notices and terms. See [NOTICE](NOTICE).
