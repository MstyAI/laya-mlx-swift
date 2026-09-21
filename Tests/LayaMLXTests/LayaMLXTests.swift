import XCTest

@testable import LayaMLX

final class LayaMLXTests: XCTestCase {
  func testQuestionTypesUseWireNames() {
    XCTAssertEqual(LayaQuestionType.choice.rawValue, "choice")
    XCTAssertEqual(LayaQuestionType.score.rawValue, "score")
    XCTAssertEqual(LayaQuestionType.boolean.rawValue, "noul")
  }

  func testPublishedCheckpoint() async throws {
    let cachedModel =
      NSHomeDirectory()
      + "/.cache/huggingface/hub/models--aac6fef--laya-mlx/snapshots/20aed815fc6acde75733882e7ec0e3f28aeb9717"
    let path = ProcessInfo.processInfo.environment["LAYA_MODEL_PATH"] ?? cachedModel
    guard FileManager.default.fileExists(atPath: path) else {
      throw XCTSkip("Set LAYA_MODEL_PATH to run checkpoint parity.")
    }
    let agent = try await LayaAgent(modelDirectory: URL(fileURLWithPath: path))
    let result = try await agent.predict(
      LayaRequest(
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
          "urgent": .boolean(instructions: "This needs urgent human attention."),
          "quality": .score(
            instructions: "Rate the customer sentiment from calm to very upset.",
            levels: ["calm", "concerned", "upset", "very upset"]
          ),
        ]
      ))
    XCTAssertEqual(result.answers["route"]?.choice, "billing")
    XCTAssertEqual(result.answers.count, 3)
    XCTAssertEqual(result.inputTokens > 0, true)
    XCTAssertEqual(result.answers["route"]?.probabilities["billing"] ?? 0, 0.9280, accuracy: 0.003)
    XCTAssertEqual(result.answers["urgent"]?.booleanProbability ?? 0, 0.3983, accuracy: 0.005)
    XCTAssertEqual(result.answers["quality"]?.score ?? 0, 2.4280, accuracy: 0.005)
    XCTAssertEqual(result.answers["route"]?.actProbability ?? 0, 1, accuracy: 0.0001)
  }
}
