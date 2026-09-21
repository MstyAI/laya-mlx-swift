import Foundation

public struct LayaRequest: Sendable {
  public let state: String
  public let questions: [String: LayaQuestion]

  public init(state: String, questions: [String: LayaQuestion]) {
    self.state = state
    self.questions = questions
  }
}

public enum LayaQuestion: Hashable, Sendable {
  case choice(instructions: String, options: [LayaOption])
  case score(instructions: String, levels: [String])
  case boolean(
    instructions: String, falseDescription: String? = nil, trueDescription: String? = nil)
}

public struct LayaOption: Hashable, Sendable {
  public let label: String
  public let description: String?

  public init(_ label: String, description: String? = nil) {
    self.label = label
    self.description = description
  }
}

public struct LayaResult: Codable, Sendable {
  public let answers: [String: LayaAnswer]
  public let inputTokens: Int
}

public struct LayaAnswer: Codable, Sendable {
  public let type: LayaQuestionType
  public let confidence: Double
  public let probabilities: [String: Double]
  public let choice: String?
  public let score: Double?
  public let booleanProbability: Double?
  public let actProbability: Double
}

public enum LayaQuestionType: String, Codable, Sendable {
  case choice
  case score
  case boolean = "noul"
}

struct PreparedQuestion: Sendable {
  let id: String
  let type: LayaQuestionType
  let tokenIDs: [Int]
  let markers: [Int]
  let labels: [String]
  let scoreLevels: [String]
}
