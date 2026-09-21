import Foundation
import LayaMLX
import XCTest

final class SupportBenchmarkTests: XCTestCase {
  func testSupportBenchmark() async throws {
    let cases = try loadSupportCases()
    let loadStart = DispatchTime.now().uptimeNanoseconds
    let agent = try await makePublishedAgent()
    let loadMilliseconds = milliseconds(since: loadStart)

    let warmupStart = DispatchTime.now().uptimeNanoseconds
    _ = try await agent.predict(makeSupportRequest(state: cases[0].state))
    let warmupMilliseconds = milliseconds(since: warmupStart)

    var samples = [SupportSample]()
    for _ in 0..<5 {
      for item in cases {
        let start = DispatchTime.now().uptimeNanoseconds
        let result = try await agent.predict(makeSupportRequest(state: item.state))
        samples.append(
          SupportSample(
            id: item.id,
            prediction: supportPrediction(from: result),
            milliseconds: milliseconds(since: start)
          ))
      }
    }

    let report = makeSupportReport(
      cases: cases,
      samples: samples,
      loadMilliseconds: loadMilliseconds,
      warmupMilliseconds: warmupMilliseconds
    )
    let data = try JSONEncoder.pretty.encode(report)
    try data.write(to: URL(fileURLWithPath: "/tmp/laya-mlx-swift-support-benchmark.json"))
    XCTAssertEqual(report.questionAccuracy, 0.74, accuracy: 0.0001)
    XCTAssertEqual(report.repeatStability, 1, accuracy: 0.0001)
  }

  private func loadSupportCases() throws -> [SupportCase] {
    let repository = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
    let data = try Data(contentsOf: repository.appending(path: "Benchmarks/support_cases.json"))
    return try JSONDecoder().decode([SupportCase].self, from: data)
  }
}

func milliseconds(since start: UInt64) -> Double {
  Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
}
