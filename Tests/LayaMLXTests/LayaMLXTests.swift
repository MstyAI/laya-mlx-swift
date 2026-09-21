import LayaMLX
import XCTest

final class LayaMLXTests: XCTestCase {
  func testQuestionTypesUseWireNames() {
    XCTAssertEqual(LayaQuestionType.choice.rawValue, "choice")
    XCTAssertEqual(LayaQuestionType.score.rawValue, "score")
    XCTAssertEqual(LayaQuestionType.boolean.rawValue, "noul")
  }

  func testPublishedCheckpoint() async throws {
    let agent = try await makePublishedAgent()
    let result = try await agent.predict(makeParityRequest())
    XCTAssertEqual(result.answers["route"]?.choice, "billing")
    XCTAssertEqual(result.answers.count, 3)
    XCTAssertGreaterThan(result.inputTokens, 0)
    XCTAssertEqual(result.answers["route"]?.probabilities["billing"] ?? 0, 0.9280, accuracy: 0.003)
    XCTAssertEqual(result.answers["urgent"]?.booleanProbability ?? 0, 0.3983, accuracy: 0.005)
    XCTAssertEqual(result.answers["quality"]?.score ?? 0, 2.4280, accuracy: 0.005)
    XCTAssertEqual(result.answers["route"]?.actProbability ?? 0, 1, accuracy: 0.0001)
  }
}
