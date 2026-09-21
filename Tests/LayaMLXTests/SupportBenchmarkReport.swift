import Foundation
import LayaMLX

struct SupportCase: Decodable {
  let id: String
  let state: String
  let expected: SupportLabels
}

struct SupportLabels: Codable, Equatable {
  let department: String
  let refund: Bool
  let urgency: Int
}

struct SupportSample: Encodable {
  let id: String
  let prediction: SupportLabels
  let milliseconds: Double
}

struct SupportReport: Encodable {
  let schemaVersion: Int
  let generatedAt: Date
  let machine: String
  let platform: String
  let modelRevision: String
  let cases: Int
  let questionsPerCase: Int
  let measuredRuns: Int
  let correctQuestions: Int
  let totalQuestions: Int
  let questionAccuracy: Double
  let exactCases: Int
  let exactCaseAccuracy: Double
  let stableRepeats: Int
  let totalRepeats: Int
  let repeatStability: Double
  let modelLoadMilliseconds: Double
  let warmupMilliseconds: Double
  let latencyMeanMilliseconds: Double
  let latencyP50Milliseconds: Double
  let latencyP95Milliseconds: Double
  let samples: [SupportSample]
}

func supportPrediction(from result: LayaResult) -> SupportLabels {
  let urgency = result.answers["urgency"]!.probabilities.max {
    $0.value == $1.value ? $0.key < $1.key : $0.value < $1.value
  }!.key
  return SupportLabels(
    department: result.answers["department"]!.choice!,
    refund: result.answers["refund"]!.booleanProbability! >= 0.5,
    urgency: Int(urgency)!
  )
}

func makeSupportReport(
  cases: [SupportCase],
  samples: [SupportSample],
  loadMilliseconds: Double,
  warmupMilliseconds: Double
) -> SupportReport {
  let firstRun = Array(samples.prefix(cases.count))
  var correctQuestions = 0
  var exactCases = 0
  for (item, sample) in zip(cases, firstRun) {
    let correct = [
      sample.prediction.department == item.expected.department,
      sample.prediction.refund == item.expected.refund,
      sample.prediction.urgency == item.expected.urgency,
    ].filter { $0 }.count
    correctQuestions += correct
    if correct == 3 { exactCases += 1 }
  }

  let baseline = Dictionary(uniqueKeysWithValues: firstRun.map { ($0.id, $0.prediction) })
  let repeats = samples.dropFirst(cases.count)
  let stable = repeats.filter { baseline[$0.id] == $0.prediction }.count
  let latencies = samples.map(\.milliseconds).sorted()
  return SupportReport(
    schemaVersion: 1,
    generatedAt: Date(),
    machine: "Apple M3 Max, 16-core CPU, 40-core GPU, 128 GB memory",
    platform: "macOS/arm64",
    modelRevision: LayaModel.defaultRevision,
    cases: cases.count,
    questionsPerCase: 3,
    measuredRuns: samples.count / cases.count,
    correctQuestions: correctQuestions,
    totalQuestions: cases.count * 3,
    questionAccuracy: Double(correctQuestions) / Double(cases.count * 3),
    exactCases: exactCases,
    exactCaseAccuracy: Double(exactCases) / Double(cases.count),
    stableRepeats: stable,
    totalRepeats: repeats.count,
    repeatStability: Double(stable) / Double(repeats.count),
    modelLoadMilliseconds: loadMilliseconds,
    warmupMilliseconds: warmupMilliseconds,
    latencyMeanMilliseconds: latencies.reduce(0, +) / Double(latencies.count),
    latencyP50Milliseconds: percentile(latencies, 0.50),
    latencyP95Milliseconds: percentile(latencies, 0.95),
    samples: samples
  )
}

private func percentile(_ sorted: [Double], _ fraction: Double) -> Double {
  sorted[Int((Double(sorted.count - 1) * fraction).rounded())]
}

extension JSONEncoder {
  static var pretty: JSONEncoder {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    encoder.dateEncodingStrategy = .iso8601
    return encoder
  }
}
